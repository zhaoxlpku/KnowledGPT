import argparse
import os
from transformers import BertConfig, BertTokenizer, BertModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-training for Knowledge-Grounded Conversation'
    )

    parser.add_argument('--out_file', type=str, default='')
    args = parser.parse_args()

    if not os.path.exists(args.out_file):
        os.makedirs(args.out_file)

    config = BertConfig.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    config.save_pretrained(args.out_file)
    tokenizer.save_pretrained(args.out_file)
    model.save_pretrained(args.out_file)
