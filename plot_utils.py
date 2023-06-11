import plotly.io as pio
# pio.renderers.default = "png"
# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.notebook as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from jaxtyping import Float, Int
from typing import List, Union, Optional
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np


def histogram(arr, title):

    mean_tqa = np.mean(arr)
    std_tqa = np.std(arr)

    # Plot histogram
    plt.hist(arr, bins=10, edgecolor='black')

    # Add vertical lines for mean and standard deviation
    plt.axvline(mean_tqa, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_tqa:.2f}')
    plt.axvline(mean_tqa - std_tqa, color='b', linestyle='dotted', linewidth=1, label=f'STD: {std_tqa:.2f}')
    plt.axvline(mean_tqa + std_tqa, color='b', linestyle='dotted', linewidth=1)

    # Set title and labels
    plt.title(title)
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')

    # Add a legend
    plt.legend()

    # Show plot
    plt.show()

def plot_head_probes(iti, head_accs, title, sort = True):
    baseline_acc = 1 - sum(iti.dataset.all_labels) / len(iti.dataset.all_labels)
    
    if sort:
        accs = -np.sort(-head_accs.reshape(iti.model.cfg.n_layers, iti.model.cfg.n_heads), axis = 1)
    else:
        accs = head_accs.reshape(iti.model.cfg.n_layers, iti.model.cfg.n_heads)
    
    fig = px.imshow(accs, labels = {"x" : "Heads", "y": "Layers"},title = title + f" Baseline Acc: {np.round(baseline_acc, 3)}", color_continuous_midpoint = baseline_acc, color_continuous_scale="RdBu", origin = "lower")
    fig.show()