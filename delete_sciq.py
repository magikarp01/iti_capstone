import os
import torch
from jaxtyping import Float, Int
from typing import List, Tuple
from torch import Tensor
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

import csv
import pandas as pd
import gc




data_dir = "/mnt/ssd-2/jamescampbell3"

inference_honest_path = "data/large_run_5/inference_outputs/inference_output_5_honest.csv"

delete_splits = ['sciq','truthfulqa']

def delete_unformatted(run_id, seq_pos, prompt_tag, act_type, splits, threshold=0, include_qa_type=[0,1],
                              d_model=8192, d_head=128, n_layers=80, n_heads=64, save_formatted=True):
    activations_dir = f"{data_dir}/data/large_run_{run_id}/activations"
    load_path = f"{activations_dir}/unformatted"
    delete_indices = []
    with open(f"{data_dir}/{inference_honest_path}", 'r') as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            if idx>0:
                ind = int(float(row[0]))
                origin_dataset = row[4]

                file_path = f"{load_path}/run_{run_id}_{prompt_tag}_{seq_pos}_{act_type}_{idx}.pt"
                file_exists = os.path.exists(file_path)
                if (file_exists) and (origin_dataset in splits):
                    os.remove(file_path)

if __name__ == "__main__":
    for act_type in tqdm(["resid_post", "resid_mid", "z", "mlp_out"]):
        for prompt_mode in tqdm(["honest", "neutral", "liar"]):
            for seq_pos in [-1,-3]:
                delete_unformatted(5, seq_pos, prompt_mode, act_type, delete_splits)
