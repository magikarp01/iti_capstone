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


mega_splits = ['sciq',
        'commonclaim',
        'creak',
        'azaria_mitchell_cities',
        'truthfulqa',
        'azaria_mitchell_capitals',
        'azaria_mitchell_companies',
        'azaria_mitchell_animals',
        'azaria_mitchell_elements',
        'azaria_mitchell_inventions',
        'azaria_mitchell_facts',
        ]

azaria_mitchell_splits = ['cities', 
                          'capitals', 
                          'companies', 
                          'animals', 
                          'elements', 
                          'inventions', 
                          'facts', 
                          'neg_companies', 
                          'neg_facts', 
                          'conj_neg_companies', 
                          'conj_neg_facts'
                          ]

data_dir = "/mnt/ssd-2/jamescampbell3"

# inference_honest_path = f"data/large_run_{run_id}/inference_outputs/inference_output_{run_id}_honest.csv"

def create_probe_dataset(run_id, seq_pos, prompt_tag, act_type, data_dir=data_dir, inference_path=None, patch_id=None, splits=mega_splits, threshold=0, include_qa_type=[0,1],
                              d_model=8192, d_head=128, n_layers=80, n_heads=64, save_formatted=True):
    """this function does both filtering for specific properties and constructs the probing dataset"""
    #assuming this runs fast, we don't need to save formatted acts, we can just format in real-time based on the property we're interested in
    if patch_id is None:
        run_dir = f"{data_dir}/data/large_run_{run_id}"
    else:
        run_dir = f"{data_dir}/data/large_run_{run_id}_patch_{patch_id}"
    activations_dir = f"{run_dir}/activations"
    load_path = f"{activations_dir}/unformatted"
    # save_path = f"{os.getcwd()}/data/large_run_5/activations/formatted"
    save_path = f"{activations_dir}/formatted"

    os.makedirs(save_path, exist_ok=True)

    probe_indices = []
    probe_labels = []    

    if inference_path is None:
        inference_path = f"inference_outputs/inference_output_{run_id}_{prompt_tag}.csv"

    # filter for the desired indices and save labels
    with open(f"{run_dir}/{inference_path}", 'r') as csvfile: #using only for meta-data but can also filter based on inf performance
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            # print(f"{idx}, {row}")
            if idx>0:
                ind = int(row[0])
                qa_type = float(row[5]) #
                origin_dataset = row[4]
                p_true = float(row[1])
                p_false = float(row[2])
                file_path = f"{load_path}/run_{run_id}_{prompt_tag}_{seq_pos}_{act_type}_{ind}.pt"
                file_exists = os.path.exists(file_path)
                if (file_exists) and (qa_type in include_qa_type) and (p_true > threshold or p_false > threshold): #and (origin_dataset in splits)
                    probe_indices.append(ind)
                    label = int(float(row[3]))
                    probe_labels.append(label)

    #create the probe dataset
    print(f"{len(probe_indices)=}")
    probe_dataset = torch.zeros((len(probe_indices), n_layers, d_model))
    probe_labels = torch.tensor(probe_labels)

    for rel_idx, base_idx in tqdm(enumerate(probe_indices)): 
        probe_dataset[rel_idx,:,:] = torch.load(f"{load_path}/run_{run_id}_{prompt_tag}_{seq_pos}_{act_type}_{base_idx}.pt")

    if save_formatted:
        if act_type in ["resid_mid", "resid_post", "mlp_out"]:
            for layer in tqdm(range(n_layers)):
                torch.save(probe_dataset[:,layer,:].squeeze().clone(), f"{save_path}/run_{run_id}_{prompt_tag}_{seq_pos}_{act_type}_{splits[0]}_l{layer}.pt") #only for single split
                torch.save(probe_labels, f"{save_path}/labels_{run_id}_{prompt_tag}_{seq_pos}_{act_type}_{splits[0]}.pt")
        elif act_type in ["z"]:
            for layer in tqdm(range(n_layers), desc='layer'):
                for head in tqdm(range(n_heads), desc='head', leave=False):
                    head_start = head*d_head
                    head_end = head_start + d_head
                    torch.save(probe_dataset[:,layer,head_start:head_end].squeeze().clone(), f"{save_path}/run_{run_id}_{prompt_tag}_{seq_pos}_{act_type}_{splits[0]}_l{layer}_h{head}.pt")
                    torch.save(probe_labels, f"{save_path}/labels_{run_id}_{prompt_tag}_{seq_pos}_{act_type}_{splits[0]}.pt")



def create_all_probe_datasets():
    for act_type in ["z", "resid_mid", "mlp_out"]:
        for mode in ["honest", "neutral", "liar"]:
            for split in mega_splits:
                create_probe_dataset(5, -1, mode, act_type, [split])
                torch.cuda.empty_cache
                gc.collect()
                print("Done with ", act_type, ", ", mode, ", ", split)


def format_logits(dataset_indices, dataset_name, run_folder, run_id, modes=["honest", "liar"], seq_pos=-1):
    #get the rows that have azaria_mitchell_facts as their value for the dataset column
    # Format logits into formatted style: run_{run_id}_{mode}_{seq_pos}_logits_{dataset_name}.pt
    inference_outputs_folder = f"{run_folder}/inference_outputs"
    formatted_folder = f"{run_folder}/activations/formatted"
        
    for mode in modes:
        logits = []
        for enum_idx, data_index in enumerate(dataset_indices):
            # with open(f"{inference_outputs_folder}/logits_{run_id}_{mode}_{data_index}.pt", "rb") as handle: change it back to this
            with open(f"{inference_outputs_folder}/logits_{run_id}_{mode}_{enum_idx}.pt", "rb") as handle: # old, replace when global indices are fixed
                logits.append(torch.load(handle))
        logits = torch.cat(logits, dim=0)
        with open(f"{formatted_folder}/run_{run_id}_{mode}_{seq_pos}_logits_{dataset_name}.pt", "wb") as handle:
            torch.save(logits, handle)