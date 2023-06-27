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
from utils.probing_utils import ModelActs
from utils.dataset_utils import CounterFact_Dataset, TQA_MC_Dataset, EZ_Dataset

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

from utils.iti_utils import patch_top_activations, patch_iti

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
random_seed = 8 # was 5 before

model.reset_hooks()
from utils.dataset_utils import MS_Dataset, MisCons_Dataset, Kinder_Dataset, HS_Dataset, EZ_Dataset

Dataset_Init = MS_Dataset

ez_data = Dataset_Init(model.tokenizer, seed=random_seed)

#%%
model.reset_hooks()
ez_acts = ModelActs(model, ez_data)
ez_acts.gen_acts(N=n_acts, id=f"ez_gpt2xl_{n_acts}")
# ez_acts.load_acts(id=f"ez_gpt2xl_{n_acts}", load_probes=False)
ez_acts.train_probes("z", max_iter=1000)

# ez_acts.save_probes(id="ez_gpt2xl_200")

#%%

cache_interventions = torch.zeros(size=(model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head))
patch_iti(model, ez_acts, use_MMD=True, cache_interventions=cache_interventions, model_device=device, alpha=10)
# patch_iti(model, ez_acts, use_probe=True, cache_interventions=cache_interventions, model_device=device, alpha=10)

# reset ez_mc so that samples will be the same
ez_data = Dataset_Init(model.tokenizer, seed=random_seed)
ez_acts_iti = ModelActs(model, ez_data)
ez_acts_iti.gen_acts(N = n_acts, id = f"iti_ez_gpt2xl_{n_acts}", indices=ez_acts.indices)
ez_acts_iti.control_for_iti(cache_interventions)

# %%
from utils.analytics_utils import plot_probe_accuracies, plot_norm_diffs, plot_cosine_sims

fig1 = plot_probe_accuracies(ez_acts)
fig1.show()
fig2 = plot_norm_diffs(ez_acts_iti, ez_acts)
fig2.show()
fig3 = plot_cosine_sims(ez_acts_iti, ez_acts)
fig3.show()
# %%
