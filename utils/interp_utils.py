import numpy as np
import torch
from collections import defaultdict
import einops
from tqdm import tqdm
import torch.nn.functional as F
from transformer_lens import HookedTransformer

def tot_logit_diff(tokenizer, model_acts, use_probs=True, eps=1e-8, test_only=True, act_type="z", check_balanced_output=False, 
                   positive_str_tokens = ["Yes", "yes", "True", "true"],
                   negative_str_tokens = ["No", "no", "False", "false"], scale_relative=False):
    """
    Get difference in correct and incorrect or positive and negative logits for each sample stored in model_acts, aggregated together.
    Should be same number of positive and negative tokens.
    If scale_relative is True, then scale probs/logits so that only correct vs incorrect or positive and negative probs/logits are considered
    """
    # positive_str_tokens = ["Yes", "yes", " Yes", " yes", "True", "true", " True", " true"]
    # negative_str_tokens = ["No", "no", " No", " no", "False", "false", " False", " false"]

    positive_tokens = [tokenizer(token).input_ids[-1] for token in positive_str_tokens]
    negative_tokens = [tokenizer(token).input_ids[-1] for token in negative_str_tokens]


    if test_only:
        sample_labels = np.array(model_acts.dataset.all_labels)[model_acts.indices_tests[act_type]] # labels
        positive_sum = torch.empty(size=(model_acts.indices_tests[act_type].shape[0],))
        negative_sum = torch.empty(size=(model_acts.indices_tests[act_type].shape[0],))
        meta_indices = np.array([np.where(model_acts.indices == i)[0][0] for i in model_acts.indices_tests["z"]])

    
    else:
        sample_labels = np.array(model_acts.dataset.all_labels)[model_acts.indices] # labels
        positive_sum = torch.empty(size=(model_acts.indices.shape[0],))
        negative_sum = torch.empty(size=(model_acts.indices.shape[0],))
        meta_indices = np.arange(model_acts.indices.shape[0],)
    
    check_positive_prop = 0

    for idx, logits in enumerate(model_acts.stored_acts["logits"][meta_indices]):

        # if answer to statement is True, correct tokens is Yes, yes, ..., true
        correct_tokens = positive_tokens if sample_labels[idx] else negative_tokens
        incorrect_tokens = negative_tokens if sample_labels[idx] else positive_tokens
        
        check_positive_prop += 1 if sample_labels[idx] else 0

        if check_balanced_output:
            correct_tokens = positive_tokens
            incorrect_tokens = negative_tokens


        if use_probs:
            probs = torch.nn.functional.softmax(logits, dim=1)
            positive_prob = probs[0, correct_tokens].sum(dim=-1)
            negative_prob = probs[0, incorrect_tokens].sum(dim=-1)

            if scale_relative:
                positive_sum[idx] = positive_prob / (positive_prob + negative_prob + eps)
                negative_sum[idx] = negative_prob / (positive_prob + negative_prob + eps)
            else:
                positive_sum[idx] = positive_prob 
                negative_sum[idx] = negative_prob 

        else:
            positive_sum[idx] = logits[0, correct_tokens].sum(dim=-1)
            negative_sum[idx] = logits[0, incorrect_tokens].sum(dim=-1)

    print(f"proportion of positive labels is {check_positive_prop/len(meta_indices)}")
    return positive_sum, negative_sum


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


def logit_attrs(model: HookedTransformer, dataset, act_types = ["resid_pre", "result"], N = 1000, indices=None, 
                positive_str_tokens=["True"], negative_str_tokens=["False"]):
    total_logit_attrs = defaultdict(list)

    if indices is None:
        indices, all_prompts, all_labels = dataset.sample(N)

    all_logits = []
    # names filter for efficiency, only cache in self.act_types
    # names_filter = lambda name: any([name.endswith(act_type) for act_type in act_types])

    for i in tqdm(indices):
        original_logits, cache = model.run_with_cache(dataset.all_prompts[i].to(model.cfg.device))


        # positive_tokens = ["Yes", "yes", " Yes", " yes", "True", "true", " True", " true"]
        # positive_str_tokens = ["Yes", "yes", "True", "true"]
        # negative_tokens = ["No", "no", " No", " no", "False", "false", " False", " false"]
        # negative_str_tokens = ["No", "no", "False", "false"]

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



def get_head_bools(model, logit_heads, flattened=False):
    """
    Method to get boolean array (n_l x n_h), 1 if head is selected at 0 if not, from a list of heads to select logit_heads.
    The flattened parameter describes the logit_heads list: if flattened is true, input to logit_heads is 1D.
    """
    if flattened:
        head_bools = torch.zeros(size=(model.cfg.total_heads,))
        for head in logit_heads:
            head_bools[head] = 1
        head_bools = einops.rearrange(head_bools, '(n_l n_h) -> n_l n_h', n_l=model.cfg.n_layers)    
    else:
        head_bools = torch.zeros(size=(model.cfg.n_layers, model.cfg.n_heads))
        for head in logit_heads:
            head_bools[head[0], head[1]] = 1
    return head_bools


def query_logits(tokenizer, logits, return_type = "logits", TOP_N = 10):
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
    top_k = [(t[1].item(), tokenizer.decode(t[0])) for t in top_k]
    
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

import torch
from torch import nn
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager

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

def get_true_false_probs(output, tokenizer, in_seq = True, scale_relative=False, eps=1e-8):
    # true_ids = [5574, 5852, 1565, 3009] #includes "true" and "True"
    # false_ids = [7700, 8824, 2089, 4541]    
    positive_str_tokens = ["Yes", "yes", "True", "true"]
    negative_str_tokens = ["No", "no", "False", "false"]
    positive_tokens = [tokenizer(token).input_ids[-1] for token in positive_str_tokens]
    negative_tokens = [tokenizer(token).input_ids[-1] for token in negative_str_tokens]
    
    if in_seq:
        output = output['logits'][:,-1,:] #last sequence position
    output = torch.nn.functional.softmax(output, dim=-1)
    # print(f"{output.shape=}, {positive_tokens=}, {negative_tokens=}")

    output = output.squeeze()
    true_prob = output[positive_tokens].sum().item()
    false_prob = output[negative_tokens].sum().item()

    if scale_relative:
        scale = (true_prob + false_prob + eps)
        true_prob /= scale
        false_prob /= scale
    return true_prob, false_prob

def batch_true_false_probs(output, tokenizer, logit_lens = True):
    positive_str_tokens = ["Yes", "yes", "True", "true"],
    negative_str_tokens = ["No", "no", "False", "false"], 
    true_ids = [tokenizer(token).input_ids[-1] for token in positive_str_tokens]
    false_ids = [tokenizer(token).input_ids[-1] for token in negative_str_tokens]

    if not logit_lens:
        output = output['logits'][:,-1,:] #last sequence position
    output = torch.nn.functional.softmax(output, dim=-1)
    # print(output.shape)
    true = output[:, true_ids].sum(axis = 1)
    false = output[:, false_ids].sum(axis = 1)
    return true, false