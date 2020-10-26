import argparse
import os
import random

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from str2bool import str2bool

from datetime import datetime

from utils import (
    save_hparams,
    KGDataset,
    collate_fn,
    get_batch_loader,
    DisBatcher,
    GenBatcher,
)
from metrics import (
    bleu_metric,
    distinct_metric,
    f1_metric
)
from model.extract import PtrExtractSumm
from model.rl import PolicyGradient
from model.util import sequence_loss, weighted_sequence_loss

from transformers import GPT2PreTrainedModel, GPT2Model, GPT2Config


class GPT2Summ(GPT2PreTrainedModel):
    '''succeed from GPT2PreTraninedModel which has implemented the 'generate' func'''

    def __init__(self, tokenizer, gpt2_config, segment=True):
        config = GPT2Config.from_pretrained(gpt2_config)
        super(GPT2Summ, self).__init__(config)
        self.transformer = GPT2Model.from_pretrained(gpt2_config)
        self.transformer.resize_token_embeddings(len(tokenizer))
        self.user_id = [tokenizer.convert_tokens_to_ids('<user1>'),
                        tokenizer.convert_tokens_to_ids('<user2>')]
        self.know_id = tokenizer.convert_tokens_to_ids('<knowledge>')
        self.segment = segment

        self.lm_head = nn.Linear(config.n_embd, len(tokenizer), bias=False)
        self.config.vocab_size = len(tokenizer)
        self.tie_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        token_type_ids = []
        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            type_ids = []
            last_special_token = self.know_id
            for j in range(len(ids)):
                if ids[j] in ([self.know_id] + self.user_id):
                    type_ids.append(ids[j])
                    last_special_token = ids[j]
                else:
                    type_ids.append(last_special_token)
            token_type_ids.append(type_ids)
        token_type_ids = torch.tensor(token_type_ids).type_as(input_ids)

        # only last token for inputs_ids if past is defined in kwargs
        if "past" in kwargs and kwargs["past"]:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        if self.segment:
            inputs = {"input_ids": input_ids, "token_type_ids": token_type_ids}
        else:
            inputs = {"input_ids": input_ids}
        inputs.update(kwargs)
        return inputs

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None):
        transformer_outputs = self.transformer(input_ids, past=past, token_type_ids=token_type_ids)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        return (lm_logits,) + transformer_outputs[1:]

    def batch_decode(self, input_ids, max_len, min_len, early_stopping, beam_size,
                     repetition_penalty, eos_id, length_penalty, no_repeat_ngram_size):
        # new-version
        output_sequences = self.generate(
            input_ids=input_ids,
            max_length=input_ids.size(1) + max_len,
            min_length=input_ids.size(1) + min_len,
            do_sample=False,
            early_stopping=early_stopping,
            num_beams=beam_size,
            repetition_penalty=repetition_penalty,
            pad_token_id=0,
            # pad_token_id=None,
            eos_token_id=eos_id,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        return output_sequences

def load_gen_net(tokenizer, segment, gpt2_config, gen_pretrain_file, load=True, cuda=True):
    gen = GPT2Summ(tokenizer=tokenizer, gpt2_config=gpt2_config, segment=segment)

    if load:
        print("Restoring all non-adagrad variables from {}...".format(gen_pretrain_file))
        state_dict = torch.load(gen_pretrain_file)['state_dict']
        gen.load_state_dict(state_dict)
    if cuda:
        gen = gen.cuda()
    return gen

def load_dis_net(emb_dim, lstm_hidden, lstm_layer, bert_config, dis_pretrain_file, load=True, cuda=True):
    dis = PtrExtractSumm(
        emb_dim=emb_dim, lstm_hidden=lstm_hidden, lstm_layer=lstm_layer, bert_config=bert_config)
    dis = PolicyGradient(dis.transformer, dis._extractor)
    if load:
        print("Restoring all non-adagrad variables from {}...".format(dis_pretrain_file))
        state_dict = torch.load(dis_pretrain_file)['state_dict']
        dis.load_state_dict(state_dict)
    if cuda:
        dis = dis.cuda()
    return dis

def main(args):
    print("\nParameters:")
    for attr, value in sorted(vars(args).items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    # Selecting wihch GPU to use
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    # Output directory for models and summaries
    out_dir = os.path.join(args.log, args.exp_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('Writing to {}\n'.format(out_dir))
    save_hparams(args, os.path.join(out_dir, 'hparams'))


    # Checkpoint directory
    checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    # Build dataset
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("Create training dataset begain... | %s " % time_str)

    test_seen_dataset = KGDataset(args.test_seen_file, max_knowledge=999)
    test_unseen_dataset = KGDataset(args.test_unseen_file, max_knowledge=999)

    test_seen_loader = get_batch_loader(test_seen_dataset, collate_fn=collate_fn, batch_size=args.eval_batch_size, is_test=True)
    test_unseen_loader = get_batch_loader(test_unseen_dataset, collate_fn=collate_fn, batch_size=args.eval_batch_size, is_test=True)

    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("Create training dataset end... | %s " % time_str)


    # Batcher
    dis_batcher = DisBatcher(args.bert_truncate, args.bert_config, args.cuda)
    gen_batcher = GenBatcher(args.knowledge_truncate, args.text_truncate, args.gpt2_truncate, args.gpt2_config, args.cuda)


    # Load model
    dis_model = load_dis_net(args.emb_dim, args.lstm_hidden, args.lstm_layer, args.bert_config, args.dis_pretrain_file, args.load_dis, args.cuda)
    gen_model = load_gen_net(gen_batcher.tokenizer, args.segment, args.gpt2_config, args.gen_pretrain_file, args.load_gen, args.cuda)


    ce = lambda logit, target: F.cross_entropy(logit, target, reduce=False)
    gen_criterion = lambda logits, targets: sequence_loss(logits, targets, ce, pad_idx=-1)

    def dev_step(split, global_step):

        if split == 'test_seen':
            test_loader = test_seen_loader
        elif split == 'test_unseen':
            test_loader = test_unseen_loader
        else:
            raise ValueError

        dis_model.eval()
        gen_model.eval()

        n_token, test_loss = 0, 0.0 # ppl
        test_hyp, test_ref = [], []
        count = 0

        with torch.no_grad():
            for knowledges, histories, users, responses, knowledge_lens in test_loader:
                knowledges = [know.split('\n\n') for know in knowledges]
                histories = [his.split('\n\n') for his in histories]

                dis_args = dis_batcher(knowledges, histories, knowledge_lens, args.n_sent)
                dis_out = dis_model(*dis_args)
                dis_knowledges = [[knowledges[bi][dis_out[0][bi].item()]] for bi in range(len(knowledges))]

                gen_args = gen_batcher(dis_knowledges, histories, users, responses, args.segment, True)
                loss = gen_criterion(gen_model(gen_args[0], token_type_ids=gen_args[1])[0], gen_args[2])
                n_token += loss.size(0)
                test_loss += loss.sum().item()

                for bi in range(len(dis_knowledges)):
                    dec_in = gen_batcher(dis_knowledges[bi:bi+1], histories[bi:bi+1], users[bi:bi+1], segment=args.segment, training=False)
                    dec_out = gen_model.batch_decode(dec_in, args.max_length, args.min_length, args.early_stopping,
                                                     args.beam_size, args.repetition_penalty, gen_batcher.eos_id,
                                                     args.length_penalty, args.no_repeat_ngram_size)
                    dec_out = dec_out[0].tolist()[dec_in.size(1):]
                    _hyp = gen_batcher.tokenizer.decode(dec_out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    _ref = responses[bi]
                    test_hyp.append(_hyp)
                    test_ref.append(_ref)

                    count += 1
                    if count % 1000 == 0:
                        print(count)

        with open(os.path.join(out_dir, '{}-decoded-iter-{}.txt'.format(split, global_step)), 'w') as f:
            for _hyp, _ref in zip(test_hyp, test_ref):
                f.writelines('{} ||| {}\n'.format(_hyp, _ref))

        MeanLoss = test_loss / n_token
        b1, b2, b3, b4 = bleu_metric(test_hyp, test_ref)
        d1, d2 = distinct_metric(test_hyp)
        f1 = f1_metric(test_hyp, test_ref)

        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("**********************************")
        print("{} results..........".format(split))
        print('hypothesis: ', len(test_hyp))
        print("Step: %d \t| ppl: %.3f \t|  %s" % (global_step, math.exp(MeanLoss), time_str))
        print("BLEU-1/2/3/4: {:.4f}/{:.4f}/{:.4f}/{:.4f}".format(b1, b2, b3, b4))
        print("Distinct-1/2: {:.4f}/{:.4f}".format(d1, d2))
        print("F1: {:.4f}".format(f1))
        print("**********************************")

        return {'f1': f1, 'loss': MeanLoss, 'bleu1': b1, 'bleu2': b2, 'bleu3': b3, 'bleu4': b4, 'distinct1': d1, 'distinct2': d2}

    dev_step("test_seen", 0)  # test_random_split
    dev_step("test_unseen", 0)  # test_topic_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-training for Knowledge-Grounded Conversation'
    )

    # files
    parser.add_argument('--test_seen_file', type=str, default='wizard_of_wikipedia/data/test_seen.jsonl')
    parser.add_argument('--test_unseen_file', type=str, default='wizard_of_wikipedia/data/test_unseen.jsonl')

    # training scheme
    parser.add_argument('--eval_batch_size', type=int, default=1)


    # save
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--log', type=str, default='wizard_of_wikipedia/log')

    parser.add_argument('--seed', type=int, default=42)

    # pre-train
    parser.add_argument('--dis_pretrain_file', type=str, default='wizard_of_wikipedia/checkpoints/model-dis-best')
    parser.add_argument('--gen_pretrain_file', type=str, default='wizard_of_wikipedia/checkpoints/model-gen-best')
    parser.add_argument('--load_dis', type=str2bool, default=True)
    parser.add_argument('--load_gen', type=str2bool, default=True)

    # model
    parser.add_argument('--bert_config', type=str, default='pretrain-models/bert_base_uncased')
    parser.add_argument('--gpt2_config', type=str, default='pretrain-models/gpt2')

    parser.add_argument('--bert_truncate', type=int, default=64) # for bert
    parser.add_argument('--gpt2_truncate', type=int, default=256) # for gpt2
    parser.add_argument('--knowledge_truncate', type=int, default=64) # for gpt2
    parser.add_argument('--text_truncate', type=int, default=64) # for gpt2
    parser.add_argument('--segment', type=str2bool, default=True)

    parser.add_argument('--n_sent', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=30)
    parser.add_argument('--min_length', type=int, default=15)
    parser.add_argument('--early_stopping', type=str2bool, default=False)
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    parser.add_argument('--emb_dim', type=int, default=768)
    parser.add_argument('--lstm_hidden', type=int, default=256)
    parser.add_argument('--lstm_layer', type=int, default=1)

    # gpu
    parser.add_argument('--gpu_list', type=str, default='2')
    parser.add_argument('--gpu_ratio', type=float, default=0.85)
    parser.add_argument('--no_cuda', type=str2bool, default=False)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True


    main(args)
