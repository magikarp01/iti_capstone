# %%
import os
import torch
from jaxtyping import Float
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


data_dir = "/mnt/ssd-2/jamescampbell3"

inference_honest_unordered_path = "data/large_run_5/inference_outputs/inference_output_5_honest_unordered.csv"
inference_honest_path = "data/large_run_5/inference_outputs/inference_output_5_honest.csv"

inference_liar_unordered_path = "data/large_run_5/inference_outputs/inference_output_5_liar_unordered.csv"
inference_liar_path = "data/large_run_5/inference_outputs/inference_output_5_liar.csv"

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

def order_inference_output(unordered_filename, ordered_filename):
    data = pd.read_csv(f"{data_dir}/{unordered_filename}")
    sorted_data = data.sort_values(by=data.columns[0])
    sorted_data.to_csv(f"{data_dir}/{ordered_filename}", index=False)


def get_inference_accuracy(filename, threshold=0, include_qa_type=[0,1], splits=mega_splits):
    num_correct = 0
    num_total = 0
    acc = 0

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            if idx>0:
                p_true = float(row[1])
                p_false = float(row[2])
                #row[4] #split
                qa_type = float(row[5]) #qa_type
                origin_dataset = row[4]
                if (p_true > threshold or p_false > threshold) and (qa_type in include_qa_type) and (origin_dataset in splits):
                    label = int(float(row[3]))
                    
                    pred = p_true > p_false
                    correct = (pred == label) #bool

                    num_correct += correct
                    num_total += 1
    if num_total > 0:
        acc = num_correct / num_total
    return acc, num_total

def get_num_labels(filename):
    num_labels = 0
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            if idx>0:
                label = int(float(row[3]))
                num_labels += label
    return num_labels

def plot_against_confidence_threshold(include_qa_type=[0,1], splits=mega_splits):
    accs_honest = []
    accs_liar = []
    totals_honest = []
    totals_liar = []
    threshs = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
    for thresh in threshs:
        acc_honest, total_honest = get_inference_accuracy(f"{data_dir}/{inference_honest_path}", threshold=thresh, include_qa_type=include_qa_type, splits=splits)
        accs_honest.append(acc_honest)
        totals_honest.append(total_honest)
        acc_liar, total_liar = get_inference_accuracy(f"{data_dir}/{inference_liar_path}", threshold=thresh, include_qa_type=include_qa_type, splits=splits)
        accs_liar.append(acc_liar)
        totals_liar.append(total_liar)

    plt.subplot(2,1,1)
    plt.plot(threshs, accs_honest, label='honest')
    plt.plot(threshs, accs_liar, label='liar')
    plt.ylabel("accuracy")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(threshs, totals_honest, label='honest')
    plt.plot(threshs, totals_liar, label='liar')
    plt.ylabel("data points")
    plt.legend()
    plt.show()


def get_accs_by_split(filename, splits=mega_splits, threshold=0, include_qa_type=[0,1]):
    accs_by_split = {}
    for split in splits:
        acc, total = get_inference_accuracy(f"{data_dir}/{filename}", splits=[split], threshold=threshold, include_qa_type=include_qa_type)
        accs_by_split[split] = (acc, total)
    return accs_by_split

