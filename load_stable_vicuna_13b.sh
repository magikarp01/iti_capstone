#!/bin/sh

sudo apt install git-lfs
git lfs install


git clone https://huggingface.co/decapoda-research/llama-13b-hf

cd ~/iti_capstone/llama-13b-hf
git lfs pull
cd ..

git clone https://huggingface.co/CarperAI/stable-vicuna-13b-delta

cd ~/iti_capstone/CarperAI/stable-vicuna-13b-delta
git lfs pull
cd ..

python ~/iti_capstone/stable-vicuna-13b-delta/apply_delta.py \
    --base ~/iti_capstone/llama-13b-hf \
    --target stable-vicuna-13b \
    --delta CarperAI/stable-vicuna-13b-delta

