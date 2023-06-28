#%%
import os
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Tuple, Callable
from functools import partial

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import csv
import gc


WORLD_SIZE = 1

device = torch.device('cuda')

#torch.set_grad_enabled(False)

model_name = "stable-vicuna-13b" #"llama-13b-hf"

batch_size = 1

num_hidden_layers = 40
hidden_size = 5120

activation_buffer = torch.zeros((num_hidden_layers, hidden_size))



@dataclass
class HookInfo:
    handle: torch.utils.hooks.RemovableHandle
    level: Optional[int] = None


class HookedModule(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._hooks: List[HookInfo] = []
        self.context_level: int = 0

    @contextmanager
    def hooks(self, fwd: List[Tuple[str, Callable]] = [], bwd: List[Tuple[str, Callable]] = []):
        self.context_level += 1
        try:
            # Add hooks
            for hook_position, hook_fn in fwd:
                module = self._get_module_by_path(hook_position)
                handle = module.register_forward_hook(hook_fn)
                info = HookInfo(handle=handle, level=self.context_level)
                self._hooks.append(info)

            for hook_position, hook_fn in bwd:
                module = self._get_module_by_path(hook_position)
                handle = module.register_full_backward_hook(hook_fn)
                info = HookInfo(handle=handle, level=self.context_level)
                self._hooks.append(info)

            yield self
        finally:
            # Remove hooks
            for info in self._hooks:
                if info.level == self.context_level:
                    info.handle.remove()
            self._hooks = [h for h in self._hooks if h.level != self.context_level]
            self.context_level -= 1

    def _get_module_by_path(self, path: str) -> nn.Module:
        module = self.model
        for attr in path.split('.'):
            module = getattr(module, attr)
        return module

    def print_model_structure(self):
        print("Model structure:")
        for name, module in self.model.named_modules():
            print(f"{name}: {module.__class__.__name__}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    

def load_llama():
    #checkpoint_location = snapshot_download(f"decapoda-research/{model_name}")
    checkpoint_location = f"{os.getcwd()}/{model_name}"
    with init_empty_weights():
        model = LlamaForCausalLM.from_pretrained(checkpoint_location)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint_location, 
        device_map="auto",
        dtype="torch.float16",
        no_split_module_classes=["LlamaDecoderLayer"],
    )

    tokenizer = LlamaTokenizer.from_pretrained(checkpoint_location)
    
    return model, tokenizer



def cache_z_hook_fnc(module, input, output, name="", layer_num=0):
    # input of shape bsz, q_len, self.hidden_size (from modeling_llama.py)
    #torch.save(input[0,-1,:].squeeze().detach(), f"{os.getcwd()}/data/example_{prompt_id}.pt")  #if tight on memory
    activation_buffer[layer_num, :] = input[0][0,-1,:].detach().clone() #input is a tuple




def main():
    model, tokenizer = load_llama()
    model.eval()

    hmodel = HookedModule(model)

    dataset = load_dataset("notrichardren/truthfulness")
    dataset = dataset["train"].remove_columns(['explanation', 'common_knowledge_label', 'origin_dataset'])

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    save_path = f"{os.getcwd()}/data"
    if not os.path.exists(save_path):
        os.system(f"mkdir {save_path}")

    hook_pairs = []
    for layer in range(model.config.num_hidden_layers):
        act_name = f"model.layers.{layer}.self_attn.o_proj"
        hook_pairs.append((act_name, partial(cache_z_hook_fnc, name=act_name, layer_num=layer)))

    for idx, batch in tqdm(enumerate(loader)):

        text = batch['claim']
        token_ids = tokenizer(text, return_tensors="pt").to(device)

        with hmodel.hooks(fwd=hook_pairs):
            output = hmodel(**token_ids)

        torch.save(activation_buffer, f"{save_path}/example_{idx}.pt")


if __name__ == "__main__":
    main()