#!/bin/sh

sudo apt-get install git-lfs
mkdir activations
git lfs install

# Stable Vicuna 13b

git clone https://huggingface.co/decapoda-research/llama-13b-hf
cd llama-13b-hf
git lfs pull
echo '{"bos_token": "", "eos_token": "", "model_max_length": 1000000000000000019884624838656, "tokenizer_class": "LlamaTokenizer", "unk_token": ""}' > ~/iti_capstone/llama-13b-hf/tokenizer_config.json
cd ..

git clone https://huggingface.co/CarperAI/stable-vicuna-13b-delta
cd stable-vicuna-13b-delta
git lfs pull
cd ..

python3 apply_delta.py --base llama-13b-hf --target stable-vicuna-13b --delta stable-vicuna-13b-delta

# Vicuna 13b

git clone https://huggingface.co/lmsys/vicuna-13b-v1.3
cd vicuna-13b-v1.3
git lfs pull
cd ..

# Vicuna 7b

git clone https://huggingface.co/decapoda-research/llama-7b-hf
cd llama-7b-hf
git lfs pull
cd ..

git clone https://huggingface.co/lmsys/vicuna-7b-delta-v1.1
cd vicuna-7b-delta-v1.1
git lfs pull
cd ..

pip install fschat
python3 -m fastchat.model.apply_delta \
    --base-model-path llama-7b-hf \
    --target-model-path vicuna-7b-hf \
    --delta-path vicuna-7b-delta-v1.1