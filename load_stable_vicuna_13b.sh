#!/bin/sh
git lfs install

git clone https://huggingface.co/decapoda-research/llama-13b-hf

cd llama-13b-hf
git lfs pull
echo '{"bos_token": "", "eos_token": "", "model_max_length": 1000000000000000019884624838656, "tokenizer_class": "LlamaTokenizer", "unk_token": ""}' > ~/iti_capstone/llama-13b-hf/tokenizer_config.json
cd ..

git clone https://huggingface.co/CarperAI/stable-vicuna-13b-delta

cd stable-vicuna-13b-delta
git lfs pull
cd ..

python stable-vicuna-13b-delta/apply_delta.py \
    --base llama-13b-hf \
    --target stable-vicuna-13b \
    --delta stable-vicuna-13b-delta

