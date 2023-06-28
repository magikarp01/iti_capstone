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

def efficient_model_loading(): #can change to cache model weights on disk
    #checkpoint_location = snapshot_download("decapoda-research/llama-30b-hf")
    checkpoint_location = f"{os.getcwd()}/vicuna-weights"
    with init_empty_weights():  # Takes up near zero memory
        model = LlamaForCausalLM.from_pretrained(checkpoint_location)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint_location,
        device_map="auto",
        dtype=torch.float16,
        no_split_module_classes=["LlamaDecoderLayer"],
    )
    tok = LlamaTokenizer.from_pretrained(checkpoint_location)
    return model, tok #returns by-reference



def save_activations(prompt, prompt_id):
    #### model loading
    #model, tok = efficient_model_loading()
    ####
    token_ids = tok(prompt, return_tensors="pt").to(model.device)
    #output = model(**token_ids)

    hmodel = HookedModule(model)
    # Print out the names modules you can add hooks to
    #hmodel.print_model_structure()

    def caching_hook_fnc(module, input, output, name=""):
        print("Hooking:", name)
        save_path = f"{os.getcwd()}/data/prompt-{prompt_id}"
        if not os.path.exists(save_path):
            os.system(f"mkdir {save_path}")
        torch.save(output[0].detach(), f"{save_path}/{name}")

    hook_pairs = []
    for layer in range(model.config.num_hidden_layers):
        for mat in ["q","k"]:
            act_name = f"model.layers.{layer}.self_attn.{mat}_proj"
            hook_pairs.append((act_name, partial(caching_hook_fnc, name=act_name)))

    with hmodel.hooks(fwd=hook_pairs):
        output = hmodel(**token_ids)

    return output
    #completion = model.generate(**token_ids, max_new_tokens=30,)
    #print(tok.batch_decode(completion)[0])

def talk(prompt, num_tokens=30):
    model, tok = efficient_model_loading()
    token_ids = tok(prompt, return_tensors="pt").to(model.device)
    completion = model.generate(**token_ids, max_new_tokens=num_tokens,)
    output = tok.batch_decode(completion)[0]
    return output

def speak(prompt, num_tokens):
    token_ids = tok(prompt, return_tensors="pt").to(model.device)
    completion = model.generate(**token_ids, max_new_tokens=num_tokens,)
    output = tok.batch_decode(completion)[0]
    print(output)


if __name__ == "__main__":
    prompt = '''Alex: "Hi, Bob!"

Bob: "Hey, Alex. How's your day going?"

Alex: "Not bad, thanks. Yours?"

Bob:'''
    num_tokens = 50
    model, tok = efficient_model_loading()
    token_ids = tok(prompt, return_tensors="pt").to(model.device)
    completion = model.generate(**token_ids, max_new_tokens=num_tokens,)
    output = tok.batch_decode(completion)[0]
    print(output)


    #prompt_id = "1"
    #output = save_activations(prompt, prompt_id)