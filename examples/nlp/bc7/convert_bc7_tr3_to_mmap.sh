#!/bin/bash

## using NeMo
# train
python preprocess_data_for_megatron.py --input=/datasets/bc7/Track-3_Med-Tweets/bc7_tr3-train.json --tokenizer-library=megatron --vocab-file=/datasets/gpt/gpt2-vocab.json --merge-file=/datasets/gpt/gpt2-merges.txt --output-prefix=/datasets/bc7/Track-3_Med-Tweets/bc7_tr3-train --tokenizer-type=gpt2

## using Megatron-LM
# train
python tools/preprocess_data.py --input=/datasets/bc7/Track-3_Med-Tweets/bc7_tr3-train.json --output-prefix=/datasets/bc7/Track-3_Med-Tweets/bc7_tr3-train --vocab=/datasets/gpt/gpt2-vocab.json --dataset-impl=mmap --tokenizer-type=GPT2BPETokenizer --merge-file=/datasets/gpt/gpt2-merges.txt

# val
python tools/preprocess_data.py --input=/datasets/bc7/Track-3_Med-Tweets/bc7_tr3-val.json --output-prefix=/datasets/bc7/Track-3_Med-Tweets/bc7_tr3-val --vocab=/datasets/gpt/gpt2-vocab.json --dataset-impl=mmap --tokenizer-type=GPT2BPETokenizer --merge-file=/datasets/gpt/gpt2-merges.txt

