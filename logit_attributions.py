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
model = HookedTransformer.from_pretrained(
    "gpt2-xl",
    center_unembed=False,
    center_writing_weights=False,
    fold_ln=False,
    refactor_factored_attn_matrices=True,
    device=device,
)
# model.to(device)
print("done")
model.set_use_attn_result(True)
model.cfg.total_heads = model.cfg.n_heads * model.cfg.n_layers

#%%
n_acts = 1000
random_seed = 5

from utils.dataset_utils import EZ_Dataset, BoolQ_Dataset, BoolQ_Question_Dataset

boolq_questions = BoolQ_Question_Dataset(model.tokenizer, seed=random_seed)

model.reset_hooks()
boolq_acts = ModelActs(model, boolq_questions, act_types=["z", "resid_pre", "result", "logits"])
# boolq_acts.gen_acts(N=n_acts, id=f"boolq_gpt2small_{n_acts}")

#%%
boolq_acts.dataset.all_prompts[0]

# %%

act_types = ["resid_pre", "result"]
total_logit_attrs = {"resid_pre": [], "result": []}

from collections import defaultdict
def logit_attrs(model: HookedTransformer, dataset, N = 1000, store_acts = True, filepath = "activations/", id = None, indices=None, storage_device="cpu"):

    if indices is None:
        indices, all_prompts, all_labels = dataset.sample(N)

    # names filter for efficiency, only cache in self.act_types
    # names_filter = lambda name: any([name.endswith(act_type) for act_type in act_types])

    for i in tqdm(indices):
        original_logits, cache = model.run_with_cache(dataset.all_prompts[i].to(model.cfg.device))
        
        positive_tokens = torch.tensor([2081, 6407, 3763, 3363])
        negative_tokens = torch.tensor([3991, 10352, 645, 1400])

        # correct_tokens = positive_tokens if dataset.all_labels[i] else negative_tokens
        # incorrect_tokens = negative_tokens if dataset.all_labels[i] else positive_tokens
        correct_tokens = " Yes" if dataset.all_labels[i] else " No"
        incorrect_tokens = " No" if dataset.all_labels[i] else " Yes"
        
        for act_type in act_types:
            stored_acts = cache.stack_activation(act_type, layer = -1).squeeze()#[:,0,-1].squeeze().to(device=storage_device)
            
            if act_type == "result":
                stored_acts = einops.rearrange(stored_acts, "n_l s n_h d_m -> (n_l n_h) s d_m")
            # print(f"{stored_acts.shape=}")
            # print(f"{cache.logit_attrs(stored_acts, tokens=correct_token, incorrect_tokens=incorrect_token)=}")
            
            total_logit_attrs[act_type].append(cache.logit_attrs(stored_acts, tokens=correct_tokens, incorrect_tokens=incorrect_tokens, pos_slice=-1, has_batch_dim=False)[:,-1]) # last position
            # print(logit_attrs)

logit_attrs(model, boolq_questions)
# %%
mean_logit_attrs = einops.rearrange(torch.stack(total_logit_attrs["result"]).mean(0), "(n_l n_h) -> n_l n_h", n_l = model.cfg.n_layers).to(device="cpu")
px.imshow(mean_logit_attrs.numpy())

#%%