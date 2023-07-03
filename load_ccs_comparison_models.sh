#!/bin/sh
#apt install git-lfs
git lfs install

git clone https://huggingface.co/microsoft/deberta-v2-xxlarge

cd deberta-v2-xxlarge
git lfs pull
cd ..

git clone https://huggingface.co/EleutherAI/gpt-j-6B
cd gpt-j-6B
git lfs pull
cd ..

git clone https://huggingface.co/t5-11b
cd t5-11b
git lfs pull
cd ..

git clone https://huggingface.co/allenai/unifiedqa-t5-11b
cd unifiedqa-t5-11b
git lfs pull
cd ..

git clone https://huggingface.co/bigscience/T0pp
cd T0pp
git lfs pull
cd ..

echo "Models downloaded and updated successfully!"