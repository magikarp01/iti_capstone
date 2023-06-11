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

from tqdm import tqdm


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
    px.imshow(utils.to_numpy(tensor), color_continuous_scale="RdBu", **kwargs).show(renderer)

def line(tensor, renderer=None, **kwargs):
    px.line(y=utils.to_numpy(tensor), **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

#%%

model = HookedTransformer.from_pretrained(
    "gpt2-xl",
    center_unembed=False,
    center_writing_weights=False,
    fold_ln=False,
    refactor_factored_attn_matrices=True,
    device=device,
)
model.set_use_attn_result(True)
model.cfg.total_heads = model.cfg.n_heads * model.cfg.n_layers

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
        
    def sample(self, sample_size: int):
        # a = list(range(len(self.all_prompts)))
        # # print(a)
        # indices = random.shuffle(a)[:sample_size]

        # indices = np.random.randint(0, len(self.dataset), size = sample_size)
        indices = np.random.choice(len(self.all_prompts), size = sample_size, replace = False)
        # print(indices) 
        # print(self.all_prompts)
        # print(self.all_labels)
        sample_prompts = []
        sample_labels =[]
        for i in indices:
            sample_prompts.append(self.all_prompts[i])
            sample_labels.append(self.all_labels[i])
        return indices, sample_prompts, sample_labels

        # print(np.array(self.all_prompts)[indices])
        # print(np.array(self.all_labels)[indices])
        # return indices, torch.tensor(self.all_prompts, )[indices], torch.tensor(self.all_labels)[indices]

#%%
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def tokenized_tqa_gen(dataset, tokenizer):
    all_prompts = []
    all_labels = []
    all_categories = []
    for i in range(len(dataset)): 
        question = dataset[i]['question']
        category = dataset[i]['category']
        rand_idx = np.random.randint(len(dataset))
        rand_question = dataset[rand_idx]['question']

        for j in range(len(dataset[i]['correct_answers'])): 
            answer = dataset[i]['correct_answers'][j]
            # prompt = format_truthfulqa_end_q(question, answer, rand_question)
            prompt = format_truthfulqa(question, answer)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(1)
            all_categories.append(category)
        
        for j in range(len(dataset[i]['incorrect_answers'])):
            answer = dataset[i]['incorrect_answers'][j]
            # prompt = format_truthfulqa_end_q(question, answer, rand_question)
            prompt = format_truthfulqa(question, answer)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(0)
            all_categories.append(category)
        
    return all_prompts, all_labels, all_categories

class TQA_GEN_Dataset():
    def __init__(self, tokenizer, category: str = "Misconceptions", seed:int = 0):
        full_dataset = load_dataset("truthful_qa", "generation")['validation']
        
        if category is None:
            self.data_set = full_dataset
        else:
            self.dataset = full_dataset.filter(lambda example: example['category'] == category)
    
        self.all_prompts, self.all_labels, self.all_categories = tokenized_tqa_gen(self.dataset, tokenizer)
        
        np.random.seed(seed)
        
    def sample(self, sample_size: int):

        indices = np.random.choice(len(self.all_prompts), size = sample_size, replace = False)

        return indices, np.array(self.all_prompts)[indices], np.array(self.all_labels)[indices]

#%%

def format_cfact(question, choice):
    return f"Q: {question} A: {choice}"

def tokenized_cfact(dataset, tokenizer): 
    all_prompts = []
    all_labels = []
    for i in range(len(dataset)):
        prompt = dataset[i]['prompt']
        target_true = dataset[i]['target_true']
        target_false = dataset[i]['target_false']

        true_prompt = prompt + target_true
        true_prompt_toks = tokenizer(true_prompt, return_tensors = 'pt').input_ids
        all_prompts.append(true_prompt_toks)
        all_labels.append(1)
        
        false_prompt = prompt + target_false
        false_prompt_toks = tokenizer(false_prompt, return_tensors = 'pt').input_ids
        all_prompts.append(false_prompt_toks)
        all_labels.append(0)
        
    return all_prompts, all_labels

class CounterFact_Dataset():
    def __init__(self, tokenizer, seed:int = 0):
        self.dataset = load_dataset("NeelNanda/counterfact-tracing")['train']
        self.all_prompts, self.all_labels = tokenized_cfact(self.dataset, tokenizer)
        
        np.random.seed(seed)
        
    def sample(self, sample_size: int):
        indices = np.random.choice(len(self.all_prompts), size = sample_size, replace = False)
        return indices, np.array(self.all_prompts)[indices], np.array(self.all_labels)[indices]

tqa_mc = TQA_MC_Dataset(model.tokenizer)

#model.tokenizer.batch_decode(tqa_mc.sample(2)[0][0])
#%%
from einops import repeat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
torch.set_grad_enabled(False)
    
# from iti import patch_top_activations

class ITI_Dataset():
    def __init__(self, model, dataset, seed = 0):
        self.model = model
        # self.model.cfg.total_heads = self.model.cfg.n_heads * self.model.cfg.n_layers
        self.dataset = dataset
        
        self.attn_head_acts = None
        self.indices = None
    
        np.random.seed(seed)
        torch.manual_seed(seed)

    def get_acts(self, N = 1000, store_acts = True, filepath = "activations/", id = None, patterns=False):
        
        attn_head_acts = []
        indices, all_prompts, all_labels = self.dataset.sample(N)
        
        patterns = []

        for i in tqdm(indices):
            original_logits, cache = model.run_with_cache(self.dataset.all_prompts[i].to(device))
            attn_head_acts.append(cache.stack_head_results(layer = -1, pos_slice = -1).squeeze(1))
            if patterns:
                for j in range(model.cfg.n_layers):
                    patterns.append(cache[utils.get_act_name("pattern", layer=j)])
        
        self.attn_head_acts = torch.stack(attn_head_acts).reshape(-1, self.model.cfg.total_heads, self.model.cfg.d_model)
        self.indices = indices
        
        if store_acts:
            if id is None:
                id = np.random.randint(10000)
            torch.save(self.indices, f'{filepath}{id}_indices.pt')
            torch.save(self.attn_head_acts, f'{filepath}{id}_attn_head_acts.pt')
            print(f"Stored at {id}")

        if patterns:
            return self.indices, self.attn_head_acts, einops.rearrange(torch.tensor(patterns), "(b n_l) ... -> b n_l ...", n_l = model.cfg.n_layers)

        return self.indices, self.attn_head_acts
        

    def load_acts(self, id, filepath = "activations/"):
        indices = torch.load(f'{filepath}{id}_indices.pt')
        attn_head_acts = torch.load(f'{filepath}{id}_attn_head_acts.pt')
        
        self.attn_head_acts = attn_head_acts
        self.indices = indices
        self.probes = None
        self.all_head_accs_np = None
        
        return indices, attn_head_acts

    def get_train_test_split(self, test_ratio = 0.2, N = None):
        attn_head_acts_list = [self.attn_head_acts[i] for i in range(self.attn_head_acts.shape[0])]
        
        indices = self.indices
        
        if N is not None:
            attn_heads_acts_list = attn_heads_acts_list[:N]
            indices = indices[:N]
        
        # print(self.attn_head_acts.shape)
        # print(len(self.dataset.all_labels))
        # print(len(indices))
        # print(np.array(self.dataset.all_labels)[indices])
        # print(len(np.array(self.dataset.all_labels)[indices]))
        
        X_train, X_test, y_train, y_test = train_test_split(attn_head_acts_list, np.array(self.dataset.all_labels)[indices], test_size=test_ratio)
        
        X_train = torch.stack(X_train, axis = 0)
        X_test = torch.stack(X_test, axis = 0)  
        
        y_train = torch.from_numpy(np.array(y_train, dtype = np.float32))
        y_test = torch.from_numpy(np.array(y_test, dtype = np.float32))
        y_train = repeat(y_train, 'b -> b num_attn_heads', num_attn_heads=self.model.cfg.total_heads)
        y_test = repeat(y_test, 'b -> b num_attn_heads', num_attn_heads=self.model.cfg.total_heads)

        return X_train, X_test, y_train, y_test

    def train_probes(self):
        X_train, X_test, y_train, y_test = self.get_train_test_split()
        print(f"{X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}")
        
        all_head_accs = []
        probes = []
        
        for i in tqdm(range(self.model.cfg.total_heads)):
                X_train_head = X_train[:,i,:]
                X_test_head = X_test[:,i,:]

                clf = LogisticRegression(max_iter=1000).fit(X_train_head.detach().numpy(), y_train[:, 0].detach().numpy())
                y_pred = clf.predict(X_train_head)
                
                y_val_pred = clf.predict(X_test_head.detach().numpy())
                all_head_accs.append(accuracy_score(y_test[:, 0].numpy(), y_val_pred))
                
                probes.append(clf)

        self.probes = probes
        
        self.all_head_accs_np = np.array(all_head_accs)
        
        return self.all_head_accs_np

#%%
i_small = torch.load("activations/tqa_gpt2xl_1000_indices.pt")
i_xl = torch.load("activations/iti_tqa_gpt2xl_1000_indices.pt")
i_xl_2 = torch.load("activations/iti_tqa_gpt2small_1000_indices.pt")

print(f"{i_small.shape}, {i_xl.shape}, {i_xl_2.shape}")

# i_small == i_xl_2
#%%

tqa_iti = ITI_Dataset(model, tqa_mc)
# _, _, orig_patterns = tqa_iti.get_acts(N=200, id = "tqa_gpt2xl_200", patterns=True)
_, _ = tqa_iti.load_acts(id = "tqa_gpt2xl_200")
tqa_all_head_accs_np = tqa_iti.train_probes()


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
    # print(f"shape of activations is {all_activations.shape}")
    # print(f"shape of truth_indices is {truth_indices.shape}")
    true_mass_mean = torch.mean(all_activations[truth_indices == 1], dim=0) #(*, d_model)
    false_mass_mean = torch.mean(all_activations[truth_indices == 0], dim=0)
    # (* d_model)

    return (true_mass_mean - false_mass_mean) / (true_mass_mean - false_mass_mean).norm()

# truthful direction is probe weight
# def get_probe_dirs(probe_list):
#     # probe is a list (n_heads len) of LogisticRegression objects
#     coefs = []
#     for probe in probe_list:
#         coefs.append(probe.coef_)
        
#     return torch.tensor(coefs, dtype=torch.float32, device=device)

def get_probe_dir(probe):
    probe_weights = torch.tensor(probe.coef_, dtype=torch.float32, device=device).squeeze()
    return probe_weights / probe_weights.norm(dim=-1)


# calculate the ITI addition (sigma * theta) for one head
# uses either MMD or probe
def calc_truth_proj(activation, use_MMD=False, use_probe=False, truth_indices=None, probe=None):
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

    # print(f"Old truthful dir direc is {truthful_dir.shape}")
    truthful_dir /= truthful_dir.norm(dim=-1)
    # print(f"New truthful dir direc is {truthful_dir.shape}")
    act_std = get_act_std(activation, truthful_dir)
    
    return einops.einsum(act_std, truthful_dir, "d_m, d_m -> d_m")

# cache after ITI caches activations after the ITI v term is added, not before
def patch_activation_hook_fn(activations, hook: HookPoint, head, old_activations, use_MMD=True, use_probe=False, truth_indices=None, probe=None, cache_after_ITI = None):
    """
    activations: (batch, n_heads, d_model)
    hook: HookPoint
    term_to_add: (*, d_model)

    A hook that is meant to act on the "z" (output) of a given head, and add the "term_to_add" on top of it. Only meant to work a certain head. Will broadcast.
    """
    # print(f"in hook fn, old act shape is {old_activations.shape}")
    term_to_add = calc_truth_proj(old_activations[:,head], use_MMD, use_probe, truth_indices, probe)
    # print(f"v shape is {term_to_add.shape}")
    # print(f"activations shape is {activations.shape}")
    # print(f"shape of cache after ITI is {cache_after_ITI.shape}")
    activations[:,-1,head] += term_to_add
    if cache_after_ITI is not None:
        cache_after_ITI[:,hook.layer(), -1,head] = activations[:,-1,head]

# Calculates new_activations for topk and adds temporary hooks
def patch_top_activations(model, probe_accuracies, old_activations, cache_after_ITI=None, topk=20, alpha=20, use_MMD=False, use_probe=False, truth_indices=None, probes=None):
    '''
    probe_accuracies: (n_layers, n_heads)
    old_activations: (batch, n_layers, n_heads, d_model)

    if use_probe is True, probes should be list in shape (n_layers, n_heads) filled with probes

    Goes into every single activation, and then tells it to add the ITI
    '''

    # print(f"old activations shape is {old_activations.shape}")

    top_head_indices = torch.topk(einops.rearrange(probe_accuracies, "n_l n_h -> (n_l n_h)"), k=topk).indices # take top k indices
    top_head_bools = torch.zeros(size=(probe_accuracies.shape[0] * probe_accuracies.shape[1],)) # set all the ones that aren't top to 0

    top_head_bools[top_head_indices] = torch.ones_like(top_head_bools[top_head_indices]) # set all the ones that are top to 1
    top_head_bools = einops.rearrange(top_head_bools, "(n_l n_h) -> n_l n_h", n_l=model.cfg.n_layers) # rearrange back


    for layer in range(probe_accuracies.shape[0]):
        for head in range(probe_accuracies.shape[1]):
            if top_head_bools[layer, head] == 1:

                if use_probe:
                    patch_activation_with_head = partial(patch_activation_hook_fn, head = head, old_activations = old_activations[:, layer], use_MMD=False, use_probe=use_probe, truth_indices=None, probe=probes[layer][head], cache_after_ITI=cache_after_ITI)
                else:
                    patch_activation_with_head = partial(patch_activation_hook_fn, head = head, old_activations = old_activations[:, layer], use_MMD=use_MMD, use_probe=False, truth_indices=truth_indices, probe=None, cache_after_ITI=cache_after_ITI)
                model.add_hook(utils.get_act_name("result", layer), patch_activation_with_head, level=-1)
    
    # return cache_after_ITI

#%%
attn_activations = tqa_iti.attn_head_acts
# activations in shape (batch, tot_heads, d_model)
truth_indices = torch.tensor(tqa_iti.dataset.all_labels)[tqa_iti.indices]

model.reset_hooks()
probe_accuracies = torch.tensor(einops.rearrange(tqa_iti.all_head_accs_np, "(n_l n_h) -> n_l n_h", n_l=model.cfg.n_layers))
old_activations = einops.rearrange(attn_activations, "b (n_l n_h) d_m -> b n_l n_h d_m", n_l=model.cfg.n_layers)

# cache_after_ITI = None

patch_top_activations(model, probe_accuracies=probe_accuracies, old_activations=old_activations, topk=50, alpha=20, use_MMD=True, truth_indices=truth_indices)

#%%
tqa_iti_2 = ITI_Dataset(model, tqa_mc)
# _, _, iti_patterns = tqa_iti_2.get_acts(N = 200, id = "BS_iti_tqa_gpt2xl_200", patterns=True)
_, _ = tqa_iti_2.load_acts(id = "BS_iti_tqa_gpt2xl_200")

#%%
norm_diffs = torch.norm((tqa_iti_2.attn_head_acts - tqa_iti.attn_head_acts), dim = 2).mean(0)
# print(cache_after_ITI.shape)
# norm_diffs = torch.norm((einops.rearrange(cache_after_ITI, "b n_l n_h d -> b (n_l n_h) d")), dim = 2).mean(0)
norm_diffs = norm_diffs.numpy().reshape(tqa_iti_2.model.cfg.n_layers, tqa_iti_2.model.cfg.n_heads)
45.39
px.imshow(norm_diffs, labels = {"x" : "Heads", "y": "Layers"},title = "Norm Differences of ITI and Normal Head Activations", color_continuous_midpoint = 0, color_continuous_scale="RdBu", origin = "lower")
#%%

import circuitsvis as cv

#%%
_, attn_patterns_cache = model.run_with_cache(tqa_iti_2.dataset.all_prompts[0])

attention_pattern = attn_patterns_cache[utils.get_act_name("pattern", layer=44)]
# attention_pattern = all_attentions[:, 45, 17]

attention_pattern = attn_patterns_cache[utils.get_act_name("pattern", layer=45)]
# attention_pattern = all_attentions[:, 45, 17]
# px.imshow(attention_pattern[0,17])

model_str_tokens = model.to_str_tokens(model.to_string(tqa_iti_2.dataset.all_prompts[0]))
print(model_str_tokens)

#%%
# px.imshow(attention_pattern[0,23])
px.imshow(attention_pattern[0,17])


# %%

from IPython.display import display

print("Layer 44 Head 23 Attention Patterns:")
attn_head_html = cv.attention.attention_heads(
    tokens=model_str_tokens,
    attention=attention_pattern,
    attention_head_names=["L44H23"],
)

#%%
import webbrowser
path = "attn_heads.html"

with open(path, "w") as f:
    f.write(str(attn_head_html))

webbrowser.open(path)

# %%
from IPython.display import display, HTML
# convert both figures to HTML divs
attn_head_html_div = attn_head_html.to_html(full_html=False)
# display the divs side by side in a HTML table
display(HTML("""
<table>
    <tr>
        <td> {div1} </td>
    </tr>
</table>
""".format(div1=attn_head_html_div)))
# %%
