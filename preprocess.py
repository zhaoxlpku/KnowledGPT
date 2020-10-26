import argparse
import os
import json
from wizard_generator import data_generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-training for Knowledge-Grounded Conversation'
    )

    parser.add_argument('--in_file', type=str, default='')
    parser.add_argument('--out_file', type=str, default='')
    args = parser.parse_args()

    # in_file = "/home/zhaoxl/Data/wizard_of_wikipedia/test_random_split.json"
    # out_file = "/home/zhaoxl/KnowledGPT/data/wizard_of_wikipedia/test_seen.jsonl"

    out_dir = os.path.dirname(args.out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.out_file, 'w', encoding='utf-8') as f:
        for history, user, response, knowledge in data_generator(args.in_file, correct_first=True, keep_last_n=2):
            f.write(
                json.dumps({
                    'history': history,
                    'user': user,
                    'response': response,
                    'knowledge': knowledge
                }) + '\n'
            )
