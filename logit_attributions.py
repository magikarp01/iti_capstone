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
from utils.probing_utils import ModelActs
from utils.dataset_utils import CounterFact_Dataset, TQA_MC_Dataset, EZ_Dataset

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

from utils.iti_utils import patch_iti

from utils.analytics_utils import plot_probe_accuracies, plot_norm_diffs, plot_cosine_sims

device = "cuda"
# %%
print("loading model")
from utils.model_utils import vicuna

model = vicuna(device = "cuda")
model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#%%
n_acts = 1000
random_seed = 5

from utils.dataset_utils import EZ_Dataset, BoolQ_Dataset, BoolQ_Question_Dataset, MS_Dataset, Kinder_Dataset, HS_Dataset

ms_questions = HS_Dataset(model.tokenizer, seed=random_seed, questions=True)

model.reset_hooks()
ms_acts = ModelActs(model, ms_questions, act_types=["z", "resid_pre", "logits"])
# boolq_acts.gen_acts(N=n_acts, id=f"boolq_gpt2small_{n_acts}")

#%%

from collections import defaultdict

def logit_attrs_tokens(cache, stored_acts, positive_tokens=[], negative_tokens=[]):
    """
    Helper function to call cache.logit_attrs over a set of possible positive and negative tokens (ints or strings). Also indexes last token. 
    Ideally, same number of positive and negative tokens (to account for relative logits)
    """
    all_attrs = []
    for token in positive_tokens:
        all_attrs.append(cache.logit_attrs(stored_acts, tokens=token, has_batch_dim=False)[:,-1])
    for token in negative_tokens:
        all_attrs.append(-cache.logit_attrs(stored_acts, tokens=token, has_batch_dim=False)[:,-1])

    return torch.stack(all_attrs).mean(0)


def logit_attrs(model: HookedTransformer, dataset, act_types = ["resid_pre", "result"], N = 1000, indices=None):
    total_logit_attrs = defaultdict(list)

    if indices is None:
        indices, all_prompts, all_labels = dataset.sample(N)

    all_logits = []
    # names filter for efficiency, only cache in self.act_types
    # names_filter = lambda name: any([name.endswith(act_type) for act_type in act_types])

    for i in tqdm(indices):
        original_logits, cache = model.run_with_cache(dataset.all_prompts[i].to(model.cfg.device))
        
        positive_tokens = torch.tensor([2081, 6407, 3763, 3363])
        negative_tokens = torch.tensor([3991, 10352, 645, 1400])


        # positive_tokens = ["Yes", "yes", " Yes", " yes", "True", "true", " True", " true"]
        positive_str_tokens = ["Yes", "yes", "True", "true"]

        # negative_tokens = ["No", "no", " No", " no", "False", "false", " False", " false"]
        negative_str_tokens = ["No", "no", "False", "false"]

        positive_tokens = [model.tokenizer(token).input_ids[-1] for token in positive_str_tokens]
        negative_tokens = [model.tokenizer(token).input_ids[-1] for token in negative_str_tokens]

        # correct_tokens = positive_tokens if dataset.all_labels[i] else negative_tokens
        # incorrect_tokens = negative_tokens if dataset.all_labels[i] else positive_tokens
        correct_tokens = positive_tokens if dataset.all_labels[i] else negative_tokens
        incorrect_tokens = negative_tokens if dataset.all_labels[i] else positive_tokens
        
        for act_type in act_types:
            stored_acts = cache.stack_activation(act_type, layer = -1).squeeze()#[:,0,-1].squeeze().to(device=storage_device)
            
            if act_type == "result":
                stored_acts = einops.rearrange(stored_acts, "n_l s n_h d_m -> (n_l n_h) s d_m")
            # print(f"{stored_acts.shape=}")
            # print(f"{cache.logit_attrs(stored_acts, tokens=correct_token, incorrect_tokens=incorrect_token)=}")
            
            # total_logit_attrs[act_type].append(cache.logit_attrs(stored_acts, tokens=correct_tokens, incorrect_tokens=incorrect_tokens, pos_slice=-1, has_batch_dim=False)[:,-1]) # last position
            total_logit_attrs[act_type].append(logit_attrs_tokens(cache, stored_acts, correct_tokens, incorrect_tokens))

        all_logits.append(original_logits)

    return all_logits, total_logit_attrs

all_logits, total_logit_attrs = logit_attrs(model, ms_questions)
# %%
mean_logit_attrs = einops.rearrange(torch.stack(total_logit_attrs["result"]).mean(0), "(n_l n_h) -> n_l n_h", n_l = model.cfg.n_layers).to(device="cpu")
px.imshow(mean_logit_attrs.numpy())

#%%
model.reset_hooks()
from utils.iti_utils import patch_iti
ms_acts.gen_acts(N=n_acts, id=f"iti_ms_llama_{n_acts}")
ms_acts.train_probes("z", max_iter=10000)
patch_iti(model, ms_acts, use_MMD=True, alpha=1, topk=1024, model_device=device)

#%%
all_logits, total_logit_attrs = logit_attrs(model, ms_questions)
mean_logit_attrs = einops.rearrange(torch.stack(total_logit_attrs["result"]).mean(0), "(n_l n_h) -> n_l n_h", n_l = model.cfg.n_layers).to(device="cpu")
px.imshow(mean_logit_attrs.numpy())

#%%
model.reset_hooks()
all_logits, total_logit_attrs = logit_attrs(model, ms_questions)
mean_logit_attrs = einops.rearrange(torch.stack(total_logit_attrs["result"]).mean(0), "(n_l n_h) -> n_l n_h", n_l = model.cfg.n_layers).to(device="cpu")
px.imshow(mean_logit_attrs.numpy())

#%%
all_logit_attrs = einops.rearrange(torch.stack(total_logit_attrs["result"]), "b (n_l n_h) -> b n_l n_h", n_l = model.cfg.n_layers).to(device="cpu")

px.histogram(all_logit_attrs[:, 26, 4])

#%%

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
