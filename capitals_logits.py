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
# %%
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

torch.set_grad_enabled(False)
device = "cuda"

# %%
model = HookedTransformer.from_pretrained(
    "gpt2-xl",
    center_unembed=False,
    center_writing_weights=False,
    fold_ln=False,
    refactor_factored_attn_matrices=True,
    device=device
)

model.cfg.total_heads = model.cfg.n_heads * model.cfg.n_layers
# %%
def query_logits(logits, return_type = "logits", TOP_N = 10):

        """
        Gets TOP_N predictions after last token in a prompt
        """
        last_tok_logits = logits[0, -1]
        
        #gets probs after last tok in seq
        
        if return_type == "probs":
            scores = F.softmax(last_tok_logits, dim=-1).detach().cpu().numpy() #the [0] is to index out of the batch idx
        else:
            scores = last_tok_logits.detach().cpu().numpy()

        #assert probs add to 1
        # assert np.abs(np.sum(probs) - 1) <= 0.01, str(np.abs(np.sum(probs)-1)) 

        probs_ = []
        for index, prob in enumerate(scores):
            probs_.append((index, prob))

        top_k = sorted(probs_, key = lambda x: x[1], reverse = True)[:TOP_N]
        top_k = [(t[1].item(), model.tokenizer.decode(t[0])) for t in top_k]
        
        return top_k
    
def is_logits_contain_label(ranked_logits, correct_answer):
    # Convert correct_answer to lower case and strip white space
    correct_answer = correct_answer.strip().lower()

    # Loop through the top 10 logits
    for logit_score, logit_value in ranked_logits:
        # Convert logit_value to lower case and strip white space
        logit_value = logit_value.strip().lower()

        # Check if the correct answer contains the logit value
        if correct_answer.find(logit_value) != -1: 
            return True
    return False
# %%
import random
import pandas as pd
from torch.utils.data import Dataset

class CapitalsDataset(Dataset):
    def __init__(self, csv_file, with_space):
        # Load the dataset
        self.dataframe = pd.read_csv(csv_file)
        self.with_space = with_space

    def __len__(self):
        # Return the length of the dataset
        return len(self.dataframe)

    def __getitem__(self, idx):
        #idx must be int
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the country and capital at the provided index
        country = self.dataframe.at[idx, 'country']
        capital = self.dataframe.at[idx, 'capital']

        # Format the input and label strings
        input_string = f"Q: What is the capital of {str(country)}? A:"
        if self.with_space:
            input_string += " \xa0"
        label_string = f"{str(capital)}"

        # Return a dict with the input and label
        sample = {'input': input_string, 'label': label_string}
        return sample

def generate_few_shot_prompt(capitals_dataset, n_few_shot, prohibited_indices=[], prop_wrong = 0):
    # Get a list of allowed indices (all indices not in prohibited_indices)
    allowed_indices = [i for i in range(len(capitals_dataset)) if i not in prohibited_indices]

    # Ensure n_few_shot is not greater than the size of the dataset
    n_few_shot = min(n_few_shot, len(allowed_indices))

    # Randomly select n_few_shot indices from the allowed indices without replacement
    indices = random.sample(allowed_indices, n_few_shot)

    # Generate the few-shot prompt
    prompt = ""
    for index in indices:
        sample = capitals_dataset[index]
        
        if np.random.rand() < prop_wrong:
            allowed_indices = [i for i in range(len(capitals_dataset)) if i not in (prohibited_indices + [index])]

            diff_idx = random.sample(allowed_indices, 1)[0]
                        
            prompt += f"{sample['input']} {capitals_dataset[diff_idx]['label']}\n"
        else:  
            prompt += f"{sample['input']} {sample['label']}\n"
        
    return prompt
# %%
# Create the original dataset with spaces
capitals_dataset_no_space = CapitalsDataset(csv_file='world_capitals.csv', with_space=False)

# Create the new dataset
few_shot_capitals_no_space_prompts = []
for i in range(len(capitals_dataset_no_space)):
    # Generate a few-shot prompt without the current index
    prompt = generate_few_shot_prompt(capitals_dataset_no_space, n_few_shot=0, prohibited_indices=[i])

    # Get the current sample and add the prompt to the 'input'
    sample = capitals_dataset_no_space[i]
    sample['input'] = prompt + sample['input']

    # Add the sample to the new dataset
    few_shot_capitals_no_space_prompts.append(sample)


# %%
few_shot_capitals_no_space_prompts[0]

# %%

# WITHOUT A SPACE
n_correct = 0 
dataset_size=300
for row in tqdm(few_shot_capitals_no_space_prompts[:dataset_size]):
    prompt = row["input"]
    label = row["label"]

    logits = model(prompt)
    
    ranked_logits = query_logits(logits, TOP_N = 1)
    
    if is_logits_contain_label(ranked_logits, label):
        n_correct +=1
        row["model_correct"] = 1
    else:
        row["model_correct"] = 0
    # print(ranked_logits)
    # print(label)
    
n_correct / len(few_shot_capitals_no_space_prompts[:dataset_size])
# %%
from dataset_utils import Capitals_Dataset
from probing_utils import ModelActs
from iti_utils import patch_iti

random_seed = 5
n_acts = 400

capitals_data = Capitals_Dataset(model.tokenizer, seed=random_seed)

capitals_acts = ModelActs(model, capitals_data)
capitals_acts.get_acts(N=n_acts, id=f"capitals_gpt2xl_{n_acts}")
# ez_acts.load_acts(id=f"ez_gpt2xl_{n_acts}", load_probes=False)
capitals_acts.train_probes(max_iter=1000)

#%%

cache_interventions = torch.zeros(size=(model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head))
patch_iti(model, capitals_acts, use_MMD=True, cache_interventions=cache_interventions, model_device=device, alpha=5, topk=20)

# reset ez_mc so that samples will be the same
capitals_data = Capitals_Dataset(model.tokenizer, seed=random_seed)
capitals_acts_iti = ModelActs(model, capitals_data)
capitals_acts_iti.get_acts(N = n_acts, id = f"iti_capitals_gpt2xl_{n_acts}", indices=capitals_acts.indices)
capitals_acts_iti.control_for_iti(cache_interventions)

# %%

# WITHOUT A SPACE
n_correct = 0 
dataset_size=300
for row in tqdm(few_shot_capitals_no_space_prompts[:dataset_size]):
    prompt = row["input"]
    label = row["label"]

    logits = model(prompt)
    
    ranked_logits = query_logits(logits, TOP_N = 1)
    
    if is_logits_contain_label(ranked_logits, label):
        n_correct +=1
        row["model_correct"] = 1
    else:
        row["model_correct"] = 0
    # print(ranked_logits)
    # print(label)
    
n_correct / len(few_shot_capitals_no_space_prompts[:dataset_size])
# %%
