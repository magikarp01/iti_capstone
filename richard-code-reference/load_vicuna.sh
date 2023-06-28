#!/bin/bash

sudo apt install git-lfs
git lfs install

git clone https://huggingface.co/decapoda-research/llama-7b-hf

cd llama-7b-hf
git lfs pull
cd ..

git clone https://huggingface.co/lmsys/vicuna-7b-delta-v1.1

cd vicuna-7b-delta-v1.1
git lfs pull
cd ..

pip install fschat

python -m fastchat.model.apply_delta \
    --base-model-path llama-7b-hf \
    --target-model-path vicuna-7b-hf \
    --delta-path vicuna-7b-delta-v1.1