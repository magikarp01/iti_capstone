import einops
import torch
import numpy as np
from functools import partial
import transformer_lens.utils as utils
from probing_utils import ModelActs
from torch.nn.functional import normalize

# Goal: define a function to take in hyperparameters Alpha and K, model probe values, and model activations to calculate the new activations

# Need to calculate alpha * sigma * theta

# theta is the truthful direction given by probe parameter

# sigma: standard deviation of head activation along truthful_dir (theta, either mass mean shift or probe weight direction)
# adapted from get_interventions_dict in Kenneth Li github


def get_act_std(head_activation, truthful_dir): # calculates standard deviations for one head
    """
    head_activation: (batch, d_head,)
    # truthful_dir: (d_head, )
    Returns standard deviation for one head of activations along truthful direction across batch
    """
    
    normalize(truthful_dir, dim=-1, out=truthful_dir)
    
    proj_act = einops.einsum(head_activation, truthful_dir , "b d_h, d_h -> b")
    return torch.std(proj_act, dim=0) # scalar

# truthful direction is difference in mean 
# returns (*, d_head)
def get_mass_mean_dir(all_activations, truth_indices): # 
    """
    all_activations: (batch, *, d_head)
    truth_indices: (batch, )
    """
    # print(f"shape of activations is {all_activations.shape}")
    # print(f"shape of truth_indices is {truth_indices.shape}")
    true_mass_mean = all_activations[truth_indices == 1].mean(dim=0) #(*, d_head)
    false_mass_mean = all_activations[truth_indices == 0].mean(dim=0)
    # (* d_head)

    return normalize(true_mass_mean - false_mass_mean, dim=-1)

# truthful direction is probe weight
# def get_probe_dirs(probe_list):
#     # probe is a list (n_heads len) of LogisticRegression objects
#     coefs = []
#     for probe in probe_list:
#         coefs.append(probe.coef_)
        
#     return torch.tensor(coefs, dtype=torch.float32, device=device)

def get_probe_dir(probe, device='cpu'):
    probe_weights = torch.tensor(probe.coef_, dtype=torch.float32, device=device).squeeze()
    return normalize(probe_weights, dim=-1)


# calculate the ITI addition (sigma * theta) for one head
# uses either MMD or probe
def calc_truth_proj(activation, use_MMD=False, use_probe=False, truth_indices=None, probe=None, device='cpu'):
    """
    activation is (batch, d_h)
    """
    if use_MMD: # use mass mean direction -- average difference between true and false classified prompts (only one head)
        assert truth_indices is not None
        truthful_dir = get_mass_mean_dir(activation, truth_indices)
    else: # probe -- just the coefficients of the probe
        assert use_probe
        assert probe is not None
        truthful_dir = get_probe_dir(probe, device=device)

    # print(f"Old truthful dir direc is {truthful_dir.shape}")
    normalize(truthful_dir, dim=-1, out=truthful_dir)
    # print(f"New truthful dir direc is {truthful_dir.shape}")
    act_std = get_act_std(activation, truthful_dir)
    
    # return einops.einsum(act_std, truthful_dir, "d_h, d_h -> d_h")
    return act_std * truthful_dir

def patch_activation_hook_fn(activations, hook, head, old_activations, alpha, use_MMD=True, use_probe=False, truth_indices=None, probe=None, cache_interventions=None, device='cpu'):
    """
    activations: (batch, n_heads, d_head)
    hook: HookPoint
    term_to_add: (*, d_head), think * is batch_size but should not exist in most cases

    cache_interventions: (n_layers, n_heads, *, d_head)

    A hook that is meant to act on the "z" (output) of a given head, and add the "term_to_add" on top of it. Only meant to work a certain head. Will broadcast.
    """
    # print(f"in hook fn, old act shape is {old_activations.shape}")
    term_to_add = calc_truth_proj(old_activations[:,head], use_MMD, use_probe, truth_indices, probe, device=device).to(device=device)
    # print(f"v shape is {term_to_add.shape}")
    # print(f"activations shape is {activations.shape}")
    
    # add to activations in last sequence position (according to paper figure 3)
    activations[:,-1,head] += alpha * term_to_add #.to(device=device)

    if cache_interventions is not None:
        cache_interventions[hook.layer(), head] = alpha * term_to_add

# Calculates new_activations for topk and adds temporary hooks
def patch_top_activations(model, probe_accuracies, old_activations, topk=20, alpha=20, use_MMD=False, use_probe=False, truth_indices=None, probes=None, cache_interventions=None, model_device='cpu'):
    """
    probe_accuracies: (n_layers, n_heads)
    old_activations: (batch, n_layers, n_heads, d_head)

    if use_probe is True, probes should be list in shape (n_layers, n_heads) filled with probes

    Goes into every single activation, and then tells it to add the ITI
    """

    # print(f"old activations shape is {old_activations.shape}")

    top_head_indices = torch.topk(einops.rearrange(probe_accuracies, "n_l n_h -> (n_l n_h)"), k=topk).indices.to(device=model_device) # take top k indices
    top_head_bools = torch.zeros(size=(probe_accuracies.shape[0] * probe_accuracies.shape[1],)).to(device=model_device) # set all the ones that aren't top to 0

    top_head_bools[top_head_indices] = torch.ones_like(top_head_bools[top_head_indices]).to(device=model_device) # set all the ones that are top to 1
    top_head_bools = einops.rearrange(top_head_bools, "(n_l n_h) -> n_l n_h", n_l=model.cfg.n_layers) # rearrange back
    
    for layer in range(probe_accuracies.shape[0]):
        for head in range(probe_accuracies.shape[1]):
            if top_head_bools[layer, head] == 1:

                if use_probe:
                    patch_activation_with_head = partial(patch_activation_hook_fn, head = head, old_activations = old_activations[:, layer], alpha=alpha, use_MMD=False, use_probe=use_probe, truth_indices=None, probe=probes[layer * model.cfg.n_heads + head], cache_interventions=cache_interventions, device=model_device)
                else:
                    patch_activation_with_head = partial(patch_activation_hook_fn, head = head, old_activations = old_activations[:, layer], alpha=alpha, use_MMD=use_MMD, use_probe=False, truth_indices=truth_indices, probe=None, cache_interventions=cache_interventions, device=model_device)
                model.add_hook(utils.get_act_name("z", layer), patch_activation_with_head)

# Do iti patching given a model and a ModelActs object
# One of use_MMD, use_probe must be true
def patch_iti(model, model_acts: ModelActs, topk=50, alpha=20, use_MMD=False, use_probe=False, cache_interventions=None, model_device='cpu'):
    assert use_MMD ^ use_probe
    model.reset_hooks()
    probe_accuracies = torch.tensor(einops.rearrange(model_acts.all_head_accs_np, "(n_l n_h) -> n_l n_h", n_l=model.cfg.n_layers)).to(device=model_device)
    
    attn_activations = model_acts.attn_head_acts
    old_activations =  einops.rearrange(attn_activations, "b (n_l n_h) d_h -> b n_l n_h d_h", n_l=model.cfg.n_layers).to(device=model_device)
    if use_MMD:
        truth_indices = torch.tensor(model_acts.dataset.all_labels)[model_acts.indices].to(device=model_device)
        probes=None
        # print(f"{attn_activations.shape=}, {probe_accuracies.shape=}, {truth_indices.shape=}")
    elif use_probe:
        probes = model_acts.probes
        truth_indices=None
    
    patch_top_activations(model, probe_accuracies, old_activations, topk=topk, alpha=alpha, use_MMD=use_MMD, use_probe=use_probe, truth_indices=truth_indices, probes=probes, cache_interventions=cache_interventions, model_device=model_device)
