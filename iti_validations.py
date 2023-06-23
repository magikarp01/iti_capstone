# Replicate ITI results, make sure ITI utils and probing utils work right

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
from probing_utils import ModelActs
from dataset_utils import CounterFact_Dataset, TQA_MC_Dataset

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

device = "cpu"
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
# import TQA Dataset

tqa_mc = TQA_MC_Dataset(model.tokenizer)
print(tqa_mc.sample(1))
# %%
model.reset_hooks()
tqa_acts = ModelActs(model, tqa_mc, seed=0)
tqa_acts.get_acts(N=200, id = "tqa_gpt2xl_200")
# tqa_acts.load_acts(id="tqa_gpt2xl_200_test", load_probes=True)
tqa_all_head_accs_np = tqa_acts.train_probes()
tqa_acts.save_probes(id="tqa_gpt2xl_200")

#%%

from iti_utils import patch_top_activations, patch_iti
patch_iti(model, tqa_acts, use_MMD=True)

# reset tqa_mc so that samples will be the same
tqa_mc = TQA_MC_Dataset(model.tokenizer, seed=0)
tqa_acts_iti = ModelActs(model, tqa_mc)
tqa_acts_iti.get_acts(N = 200, id = "iti_tqa_gpt2xl_200", indices=tqa_acts.indices)
# tqa_iti_2.load_acts(id = "iti_tqa_gpt2xl_200")
# %%

probe_accuracies = torch.tensor(einops.rearrange(tqa_acts.all_head_accs_np, "(n_l n_h) -> n_l n_h", n_l=model.cfg.n_layers))
top_head_indices = torch.topk(einops.rearrange(probe_accuracies, "n_l n_h -> (n_l n_h)"), k=50).indices # take top k indices
top_head_indices

top_probe_heads = torch.zeros(size=(1200,))
top_probe_heads[top_head_indices] = 1
top_probe_heads = einops.rearrange(top_probe_heads, "(n_l n_h) -> n_l n_h", n_l=model.cfg.n_layers)
for l in range(model.cfg.n_layers):
    for h in range(model.cfg.n_heads):
        if top_probe_heads[l, h] == 1:
            print(f"layer {l}, head {h}")

#%%


tqa = -np.sort(-tqa_acts.all_head_accs_np.reshape(tqa_acts.model.cfg.n_layers, tqa_acts.model.cfg.n_heads), axis = 1)
px.imshow(tqa, labels = {"x" : "Heads (sorted)", "y": "Layers"},title = "Probe Accuracies TQA", color_continuous_midpoint = 0.5, color_continuous_scale="YlGnBu", origin = "lower")

#%%

norm_diffs = torch.norm((tqa_acts_iti.attn_head_acts - tqa_acts.attn_head_acts), dim = 2).mean(0)

norm_diffs = norm_diffs.numpy().reshape(tqa_acts_iti.model.cfg.n_layers, tqa_acts_iti.model.cfg.n_heads)
px.imshow(norm_diffs, labels = {"x" : "Heads", "y": "Layers"},title = "Norm Differences of ITI and Normal Head Activations", color_continuous_midpoint = 0, color_continuous_scale="RdBu", origin = "lower")

#%%
norm_diffs = torch.norm((tqa_acts_iti.attn_head_acts - tqa_acts.attn_head_acts), dim = 2).mean(0) / torch.norm(tqa_acts.attn_head_acts, dim = 2).mean(0)
norm_diffs = norm_diffs.numpy().reshape(tqa_acts.model.cfg.n_layers, tqa_acts.model.cfg.n_heads)

px.imshow(norm_diffs, labels = {"x" : "Heads", "y": "Layers"},title = "Norm Differences (divided by original norm) of ITI and Normal Head Activations", color_continuous_midpoint = 0, color_continuous_scale="RdBu", origin = "lower")

#%%
act_sims = torch.nn.functional.cosine_similarity(tqa_acts_iti.attn_head_acts, tqa_acts.attn_head_acts, dim=2).mean(0)
act_sims = act_sims.numpy().reshape(tqa_acts_iti.model.cfg.n_layers, tqa_acts_iti.model.cfg.n_heads)

# act_sims[44, 23] = act_sims[45, 17] = 1
px.imshow(act_sims, labels = {"x" : "Heads", "y": "Layers"},title = "Cosine Similarities of of ITI and Normal Head Activations", color_continuous_midpoint = 1, color_continuous_scale="RdBu", origin = "lower")


#%%
model.reset_hooks()

cfact = CounterFact_Dataset(model.tokenizer)
print(cfact.sample(1))

cfact_acts = ModelActs(model, cfact)
cfact_acts.get_acts(N=200, id="cfact_gpt2xl_200")
# cfact_acts.load_acts(id="cfact_gpt2xl_200")
cfact_all_head_accs_np = cfact_acts.train_probes()

#%%
cfact = CounterFact_Dataset(model.tokenizer)
print(cfact.sample(1))

from iti_utils import patch_top_activations, patch_iti
patch_iti(model, cfact_acts, use_MMD=True)

cfact_acts_iti = ModelActs(model, cfact, indices=cfact_acts.indices)
cfact_acts_iti.get_acts(N = 200, id = "iti_cfact_gpt2xl_200")


# %%
norm_diffs = torch.norm((cfact_acts_iti.attn_head_acts - cfact_acts.attn_head_acts), dim = 2).mean(0) / torch.norm(cfact_acts.attn_head_acts, dim = 2).mean(0)
norm_diffs = norm_diffs.numpy().reshape(cfact_acts.model.cfg.n_layers, cfact_acts.model.cfg.n_heads)

px.imshow(norm_diffs, labels = {"x" : "Heads", "y": "Layers"},title = "Norm Differences (divided by original norm) of ITI and Normal Head Activations", color_continuous_midpoint = 0, color_continuous_scale="RdBu", origin = "lower")

#%%
act_sims = torch.nn.functional.cosine_similarity(cfact_acts_iti.attn_head_acts, cfact_acts.attn_head_acts, dim=2).mean(0)
act_sims = act_sims.numpy().reshape(cfact_acts_iti.model.cfg.n_layers, cfact_acts_iti.model.cfg.n_heads)

# act_sims[44, 23] = act_sims[45, 17] = 1
px.imshow(act_sims, labels = {"x" : "Heads", "y": "Layers"},title = "Cosine Similarities of of ITI and Normal Head Activations", color_continuous_midpoint = 1, color_continuous_scale="RdBu", origin = "lower")

#%%

model.reset_hooks()
from dataset_utils import EZ_Dataset

ez_data = EZ_Dataset(model.tokenizer)
print(ez_data.sample(1))

#%%
ez_acts = ModelActs(model, ez_data)
ez_acts.get_acts(N=200, id="ez_gpt2xl_200")
# cfact_acts.load_acts(id="cfact_gpt2xl_200")
ez_acts.train_probes()

ez_acts.save_probes(id="ez_gpt2xl_200")

# %%
ez = -np.sort(-ez_acts.all_head_accs_np.reshape(ez_acts.model.cfg.n_layers, ez_acts.model.cfg.n_heads), axis = 1)
px.imshow(ez, labels = {"x" : "Heads (sorted)", "y": "Layers"},title = "Probe Accuracies TQA", color_continuous_midpoint = 0.5, color_continuous_scale="YlGnBu", origin = "lower")

#%%
patch_iti(model, ez_acts, use_MMD=True)

# reset tqa_mc so that samples will be the same
ez_data = EZ_Dataset(model.tokenizer, seed=0)
ez_acts_iti = ModelActs(model, ez_data)
ez_acts_iti.get_acts(N = 200, id = "iti_ez_gpt2xl_200", indices=ez_acts.indices)

# %%
from analytics_utils import plot_probe_accuracies, plot_norm_diffs, plot_cosine_sims

fig1 = plot_probe_accuracies(ez_acts)
fig1.show()
fig2 = plot_norm_diffs(ez_acts_iti, ez_acts)
fig2.show()
fig3 = plot_cosine_sims(ez_acts_iti, ez_acts)
fig3.show()

# %%
