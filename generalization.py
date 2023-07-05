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




# %%
device = "cuda"
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

model.reset_hooks()



# %%
random_seed = 5

datanames = ["tqa", "cfact", "ez"]

tqa_data = TQA_MC_Dataset(model.tokenizer, seed=random_seed)
cfact_data = CounterFact_Dataset(model.tokenizer, seed=random_seed)
ez_data = EZ_Dataset(model.tokenizer, seed=random_seed)


datasets = {"tqa":tqa_data, "cfact":cfact_data, "ez":ez_data}



# %%
n_acts = 1000
acts = {}

for name in datanames:
    acts[name] = ModelActs(model, datasets[name], act_types=["z", "mlp_out", "resid_post", "resid_pre", "result"])
    model_acts: ModelActs = acts[name]
    model_acts.gen_acts(N=n_acts, id=f"{name}_gpt2xl_{n_acts}")
    # break
    # ez_acts.load_acts(id=f"ez_gpt2xl_{n_acts}", load_probes=False)
    model_acts.train_probes(max_iter=1000)

# %%
from plotly.subplots import make_subplots
from utils.gpt_judge import check_iti_generalization
plots = []

for name in datanames:
    model_acts: ModelActs = acts[name]
    for other_name in datanames:
        print(f"Checking generation on {name}, ITI on {other_name}")
        results = check_iti_generalization(model, datasets[name], datasets[other_name], 50, 1000, alpha=10)
        print(f"Truth score before ITI: {results[0]}, Truth score after ITI: {results[2]}")
        print(f"Info score before ITI: {results[1]}, Info score after ITI: {results[3]}")
        print()

        # transfer_accs = model_acts.get_transfer_acc(acts[other_name])
        # plots.append(plot_probe_accuracies(model_acts, sorted=False, title=f"{name} probes on {other_name} data", other_head_accs=transfer_accs).show())


# %%
from plotly.subplots import make_subplots
from utils.gpt_judge import check_iti_generalization
plots = []

np.seterr(all="ignore")
for name in datanames[1:]:
    model_acts: ModelActs = acts[name]
    for other_name in datanames:
        print(f"Checking generation on {name}, ITI on {other_name}")
        results = check_iti_generalization(model, datasets[name], datasets[other_name], 50, 1000, alpha=10, existing_gen_acts=acts[name])
        print(f"Truth score before ITI: {results[0]}, Truth score after ITI: {results[2]}")
        print(f"Info score before ITI: {results[1]}, Info score after ITI: {results[3]}")
        print()

        # transfer_accs = model_acts.get_transfer_acc(acts[other_name])
        # plots.append(plot_probe_accuracies(model_acts, sorted=False, title=f"{name} probes on {other_name} data", other_head_accs=transfer_accs).show())