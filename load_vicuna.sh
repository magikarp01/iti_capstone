#!/bin/bash

sudo apt install git-lfs
git lfs install

git clone https://huggingface.co/decapoda-research/llama-7b-hf

cd ~/rlhf-truthfulness/llama-7b-hf
git lfs pull
cd ..

git clone https://huggingface.co/lmsys/vicuna-7b-delta-v1.1

cd ~/rlhf-truthfulness/vicuna-7b-delta-v1.1
git lfs pull
cd ..

pip install fschat

python -m fastchat.model.apply_delta \
    --base-model-path ~/rlhf-truthfulness/llama-7b-hf \
    --target-model-path ~/rlhf-truthfulness/vicuna-7b-hf \
    --delta-path ~/rlhf-truthfulness/vicuna-7b-delta-v1.1