#%%
# Janky code to do different setup when run in a Colab notebook vs VSCode
DEVELOPMENT_MODE = False

IN_COLAB = False
print("Running as a Jupyter notebook - intended for development only!")
from IPython import get_ipython

ipython = get_ipython()
# Code to automatically update the HookedTransformer code as its edited without restarting the kernel
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")
#%%
# Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
import plotly.io as pio
if IN_COLAB or not DEVELOPMENT_MODE:
    pio.renderers.default = "colab"
else:
    pio.renderers.default = "notebook_connected"
print(f"Using renderer: {pio.renderers.default}")

import circuitsvis as cv
#%%
# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
from tqdm import tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from torchtyping import TensorType as TT
from typing import List, Union, Optional
from jaxtyping import Float, Int
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML
# import circuitsvis as cv

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

torch.set_grad_enabled(False)

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)
# %%
from transformers import LlamaForCausalLM, LlamaTokenizer

# TODO
MODEL_PATH='llama_model/hf_llama_model'

tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
hf_model = LlamaForCausalLM.from_pretrained(MODEL_PATH, low_cpu_mem_usage = True, torch_dtype=torch.float32, device_map = "auto")
print(torch.cuda.memory_allocated())

#%%
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate
generate_ids = hf_model.generate(inputs.input_ids, max_length=60)

tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# tokenizer.decode(generate_ids[0])
# %%


model = HookedTransformer.from_pretrained("llama-7b", hf_model=hf_model, device="cpu", fold_ln=False, center_writing_weights=False, center_unembed=False)
print(torch.cuda.memory_allocated())


model.tokenizer = tokenizer
model.tokenizer.add_special_tokens({"pad_token": "[PAD]"})


#%%

model.generate("The capital of Germany is", max_new_tokens=20, temperature=0)

# %% 
# HookedTransformer.from_pretrained("llama-7b", hf_model=hf_model, device="cuda", fold_ln=False, center_writing_weights=False, center_unembed=False)
