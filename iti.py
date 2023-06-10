# Main goal: Run hyperparameter sweep over ITI given the probe values

#%%
from IPython import get_ipython

ipython = get_ipython()
# Code to automatically update the TransformerLens code as its edited without restarting the kernel
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")
    
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

#%%
# import pysvelte
# 
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

device = "cpu"

#%%

def imshow(tensor, renderer=None, **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", **kwargs).show(renderer)

def line(tensor, renderer=None, **kwargs):
    px.line(y=utils.to_numpy(tensor), **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)
    
#%%

model = HookedTransformer.from_pretrained(
    "gpt2-medium",
    center_unembed=False,
    center_writing_weights=False,
    fold_ln=False,
    refactor_factored_attn_matrices=True,
    device=device
)

NUM_HEADS = 384

#%%

from datasets import load_dataset
from sklearn.model_selection import train_test_split

def format_truthfulqa(question, choice):
    return f"Q: {question} A: {choice}"

def tokenized_tqa(dataset, tokenizer): 

    all_prompts = []
    all_labels = []
    for i in range(len(dataset)):
        question = dataset[i]['question']
        choices = dataset[i]['mc2_targets']['choices']
        labels = dataset[i]['mc2_targets']['labels']

        assert len(choices) == len(labels), (len(choices), len(labels))

        for j in range(len(choices)): 
            choice = choices[j]
            label = labels[j]
            prompt = format_truthfulqa(question, choice)
            if i == 0 and j == 0: 
                print(prompt)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(label)
    
    return all_prompts, all_labels

class TQA_MC_Dataset():
    def __init__(self, tokenizer, seed:int = 0):
        self.dataset = load_dataset("truthful_qa", "multiple_choice")['validation']
        self.all_prompts, self.all_labels = tokenized_tqa(self.dataset, tokenizer)
        
        np.random.seed(seed)

    # def get_train_test_split(self, test_ratio = 0.2):
    #     X_train, X_test, y_train, y_test = train_test_split(self.all_prompts, self.all_labels, test_size=test_ratio)
    #     return X_train, X_test, y_train, y_test
        
    def sample(self, sample_size: int):
        indices = np.random.randint(0, len(self.dataset), size = sample_size)
        return self.all_prompts[indices], self.all_labels[indices]

#%%

tqa_mc = TQA_MC_Dataset(model.tokenizer)

#model.tokenizer.batch_decode(tqa_mc.sample(2)[0][0])

#%%

# class ITI_Dataset():
#     def __init__(self, model, dataset):
#         self.model = model
#         self.dataset = dataset
    
#     # def get_acts():

# #%%

# X_train, X_test, y_train, y_test = tqa_mc.get_train_test_split()

#%%

from tqdm import tqdm
# Run the model and cache all activations

# torch.from_numpy(tqa_mc.all_prompts).to(device)
N = 100
attn_head_acts = []
for i in tqdm(range(N)):
    original_logits, cache = model.run_with_cache(tqa_mc.all_prompts[i].to(device))
    
    attn_head_acts.append(cache.stack_head_results(layer = -1, pos_slice = -1).squeeze(1))

#%%

X_train, X_test, y_train, y_test = train_test_split(attn_head_acts, tqa_mc.all_labels[:N], test_size=0.2)

#%%

X_train = torch.stack(X_train, axis = 0)
X_test = torch.stack(X_test, axis = 0)

#%%

sum(y_train) + sum(y_test)

#%%

y_train = torch.from_numpy(np.array(y_train, dtype = np.float32))
y_test = torch.from_numpy(np.array(y_test, dtype = np.float32))

#%%

from einops import repeat

y_train = repeat(y_train, 'b -> b num_attn_heads', num_attn_heads=NUM_HEADS)
y_test = repeat(y_test, 'b -> b num_attn_heads', num_attn_heads=NUM_HEADS)

#%%

import torch
from torch import nn
from torch.optim import Adam

class LinearProbes(nn.Module):
    def __init__(self, num_probes, input_dim):
        super().__init__()
        self.probes = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for _ in range(num_probes)])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is of shape (batch_size, num_probes, input_dim)
        # we apply each probe to its corresponding attention head output
        out = [self.sigmoid(probe(x[:, i])) for i, probe in enumerate(self.probes)]
        return torch.stack(out, dim=1)  # shape: (batch_size, num_probes, 1)

torch.set_grad_enabled(True)

# Initialize model and optimizer
model = LinearProbes(num_probes=NUM_HEADS, input_dim=1024) # 1024 is d_model
optimizer = Adam(model.parameters())
loss_fn = nn.BCELoss()
# Assume X_train and y_train are tensors
# X_train = torch.from_numpy(X_train)
# y_train = torch.from_numpy(y_train)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):  # num_epochs is the number of training epochs
    optimizer.zero_grad()
    
    # forward pass
    outputs = model(X_train).squeeze()  # shape: (batch_size, num_probes)

    # compute loss
    loss = loss_fn(outputs, y_train)

    # backward pass
    loss.backward()

    # update weights
    optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

#%%

# Get the model's predictions on the test set
with torch.no_grad():
    test_outputs = model(X_test).squeeze()  # shape: (batch_size, num_probes)

# Round the outputs to get the class predictions
test_preds = torch.round(test_outputs)

# Compute the accuracy
correct_preds = (test_preds == y_test).float()  # convert boolean tensor to float
acc_per_probe = correct_preds.mean(axis = 0)
overall_accuracy = correct_preds.mean().item()

print(f'Accuracy: {overall_accuracy * 100}%')
print(f'Max Probe Accuracy: {acc_per_probe.max() * 100}%')
print(f'Min Probe Accuracy: {acc_per_probe.min() * 100}%')
#%%
acc_per_probe.numpy()
#%%
import matplotlib.pyplot as plt

# Example data

plt.hist(acc_per_probe.numpy(), bins=10, edgecolor='black')  # Choose 10 bins

plt.title('Distribution of Probe accuracies (80 train samples, 20 test samples)')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')

plt.show()
#%%
# Goal: define a function to take in hyperparameters Alpha and K, model probe values, and model activations to calculate the new activations

# Need to calculate alpha * sigma * theta

# theta is the truthful direction given by probe parameter

# sigma: standard deviation of head activation along truthful_dir (theta, either mass mean shift or probe weight direction)
# adapted from get_interventions_dict in Kenneth Li github
'''
def get_act_std(head_activations, truthful_dirs):
    # head_activation: (batch, n_heads, d_model,)
    # truthful_dirs: (n_heads, d_model,)
    truthful_dirs /= torch.norm(truthful_dirs, dim=-1, keepdim=True)
    proj_act = einops.einsum(head_activations, truthful_dirs , "b n_h d_m, n_h d_m -> b n_h d_m")
    return torch.std(proj_act, dim=0) # (n_h d_m)
'''
def get_act_std(head_activation, truthful_dir): # calculates standard deviations for one head
    """
    head_activation: (batch, d_model,)
    # truthful_dir: (d_model, )
    """
    truthful_dir /= torch.norm(truthful_dir, dim=-1, keepdim=True)
    proj_act = einops.einsum(head_activation, truthful_dir , "b d_m, d_m -> b d_m")
    return torch.std(proj_act, dim=0) # (d_m)

# truthful direction is difference in mean 
# returns (*, d_model)
def get_mass_mean_dir(all_activations, truth_indices): # 
    """
    all_activations: (batch, *, d_model)
    truth_indices: (batch, )
    """
    true_mass_mean = torch.mean(all_activations[truth_indices == 1], dim=0) #(*, d_model)
    false_mass_mean = torch.mean(all_activations[truth_indices == 0], dim=0)
    # (* d_model)

    return true_mass_mean - false_mass_mean

# truthful direction is probe weight
# def get_probe_dirs(probe_list):
#     # probe is a list (n_heads len) of LogisticRegression objects
#     coefs = []
#     for probe in probe_list:
#         coefs.append(probe.coef_)
        
#     return torch.tensor(coefs, dtype=torch.float32, device=device)

def get_probe_dir(probe):
    return torch.tensor(probe.coefs, dtype=torch.float32, device=device)


# calculate the ITI addition (sigma * theta) for one head
# uses either MMD or probe
def calc_truth_proj(activation, use_MMD=True, use_probe=False, truth_indices=None, probe=None):
    '''
    activation is (batch, d_m)
    '''
    if use_MMD: # use mass mean direction -- average difference between true and false classified prompts (only one head)
        assert truth_indices is not None
        truthful_dir = get_mass_mean_dir(activation, truth_indices)
    else: # probe -- just the coefficients of the probe
        assert use_probe
        assert probe is not None
        truthful_dir = get_probe_dir(probe)
    
    act_std = get_act_std(activation, truthful_dir)
    
    return einops.einsum(act_std, truthful_dir, "d_m, d_m -> d_m")

def patch_activation(activations, hook: HookPoint, head, use_MMD=True, use_probe=False, truth_indices=None, probe=None):
    """
    activations: (batch, n_heads, d_model)
    hook: HookPoint
    term_to_add: (*, d_model)

    A hook that is meant to act on the "z" (output) of a given head, and add the "term_to_add" on top of it. Only meant to work a certain head. Will broadcast.
    """
    term_to_add = calc_truth_proj(activations[:,head], use_MMD, use_probe, truth_indices, probe)
    return activations[:,head,:] + term_to_add

# Calculates new_activations for topk and adds temporary hooks
def patch_top_activations(model, probe_accuracies, topk=20, alpha=20, use_MMD=True, use_probe=False, truth_indices=None, probes=None):
    '''
    orig_activations: (batch, n_layers, n_heads, d_model)
    truthful_dirs: (n_layers, n_heads, d_model)
    probe_accuracies: (n_layers, n_heads)

    if use_probe is True, probes should be list in shape (n_layers, n_heads) filled with probes

    Goes into every single activation, and then tells it to add the ITI
    '''

    top_head_indices = torch.topk(einops.rearrange(probe_accuracies, "n_l n_h -> (n_l n_h)"), k=topk) # take top k indices
    top_head_bools = torch.zeros(size=(probe_accuracies.shape[0] * probe_accuracies.shape[1])) # set all the ones that aren't top to 0
    top_head_bools[top_head_indices] = torch.tensor(1) # set all the ones that are top to 1
    top_head_bools = einops.rearrange(top_head_bools, "(n_l n_h) -> n_l n_h") # rearrange back
    
    for layer in range(probe_accuracies.shape[0]):
        for head in range(probe_accuracies.shape[1]):
            if top_head_bools[layer, head] == 1:

                if use_probe:
                    patch_activation_with_head = partial(patch_activation, head = head, use_MMD=False, use_probe=use_probe, truth_indices=None, probe=probes[layer][head])
                else:
                    patch_activation_with_head = partial(patch_activation, head = head, use_MMD=use_MMD, use_probe=False, truth_indices=truth_indices, probe=None)
                model.add_hook((utils.get_act_name("attn_out", layer), patch_activation_with_head))
    
    # orig_top_acts = orig_activations[:, top_head_indices] # (batch, topk, d_model)
    # top_truthful_dirs = truthful_dirs[top_head_indices] # (topk, d_model)

    # ITI_adds = torch.zeros_like(orig_top_acts)
    
    # # iterate through every top head
    # for head_index in range(topk):
    #     truthful_dir = top_truthful_dirs[head_index] # (d_model)
    #     orig_act = orig_top_acts[:, head_index] # (batch, d_model)
    #     act_std = get_act_std(orig_act, truthful_dir) # (d_model)

    #     ITI_adds[:, head_index] = alpha * einops.einsum(act_std, truthful_dir, "d_m, d_m -> d_m")


