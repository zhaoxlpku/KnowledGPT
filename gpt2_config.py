import argparse
import os
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-training for Knowledge-Grounded Conversation'
    )

    parser.add_argument('--out_file', type=str, default='')
    args = parser.parse_args()

    if not os.path.exists(args.out_file):
        os.makedirs(args.out_file)

    config = GPT2Config.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')

    config.save_pretrained(args.out_file)
    tokenizer.save_pretrained(args.out_file)
    model.save_pretrained(args.out_file)
