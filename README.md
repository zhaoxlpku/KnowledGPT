# KnowledGPT

This is an implementation of our paper
- Xueliang Zhao, Wei Wu, Can Xu, Chongyang Tao, Dongyan Zhao and Rui Yan. Knowledge-Grounded Dialogue Generation with Pre-trained Language Models. In *EMNLP*, 2020 (Main Conference) 

## Requirements

- Python 3.6
- Pytorch 1.2.0
- CUDA 10.0 supported GPU with at least 12GB memory
- see [requirements.txt](requirements.txt) for more details

## Usage

To run our pretrained model on Wizard of Wikipedia:

- Download the data from [ParlAI](http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz), then preprocess the data
```bash
python preprocess.py --in_file /path/to/wizard_of_wikipedia/test_random_split.json --out_file wizard_of_wikipedia/data/test_seen.jsonl
python preprocess.py --in_file /path/to/wizard_of_wikipedia/test_topic_split.json --out_file wizard_of_wikipedia/data/test_unseen.jsonl
```

- Prepare the BERT/GPT-2 files
```bash
python bert_config.py --out_file pretrain-models/bert_base_uncased
python gpt2_config.py --out_file pretrain-models/gpt2
```

- Download the checkpoint from [here](https://drive.google.com/file/d/1SFRrU9Yu_smbDNzlRPcc6gnOIAtj1dzk/view?usp=sharing), then save to "wizard_of_wikipedia/checkpoints"

- To evaluate the model, run
```bash
python evaluate.py --eval_batch_size 2 --gpu_list 0 --exp_name test
```


## Citation
We appreciate your citation if you find our code is beneficial.

```
@inproceedings{zhao2020knowledgpt,
  title={Knowledge-Grounded Dialogue Generation with Pre-trained Language Models},
  author={Xueliang Zhao, Wei Wu, Can Xu, Chongyang Tao, Dongyan Zhao, Rui Yan},
  booktitle = {EMNLP},
  year = {2020}
}
```