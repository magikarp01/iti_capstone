import einops
import torch
import numpy as np

# Goal: define a function to take in hyperparameters Alpha and K, model probe values, and model activations to calculate the new activations

# Need to calculate alpha * sigma * theta

# theta is the truthful direction given by probe parameter

# sigma: standard deviation of head activation along truthful_dir (theta, either mass mean shift or probe weight direction)
# adapted from get_interventions_dict in Kenneth Li github

def get_act_std(head_activation, truthful_dir): # calculates standard deviations for one head
    """
    head_activation: (batch, d_model,)
    # truthful_dir: (d_model, )
    """
    truthful_dir /= torch.norm(truthful_dir, dim=-1, keepdim=True)
    proj_act = einops.einsum(head_activation, truthful_dir , "b d_m, d_m -> b d_m")
    return torch.std(proj_act, dim=0) # (d_m)

# truthful direction is difference in mean 
# returns (*, d_model)
def get_mass_mean_dir(all_activations, truth_indices): # 
    """
    all_activations: (batch, *, d_model)
    truth_indices: (batch, )
    """
    # print(f"shape of activations is {all_activations.shape}")
    # print(f"shape of truth_indices is {truth_indices.shape}")
    true_mass_mean = torch.mean(all_activations[truth_indices == 1], dim=0) #(*, d_model)
    false_mass_mean = torch.mean(all_activations[truth_indices == 0], dim=0)
    # (* d_model)

    return (true_mass_mean - false_mass_mean) / (true_mass_mean - false_mass_mean).norm()

# truthful direction is probe weight
# def get_probe_dirs(probe_list):
#     # probe is a list (n_heads len) of LogisticRegression objects
#     coefs = []
#     for probe in probe_list:
#         coefs.append(probe.coef_)
        
#     return torch.tensor(coefs, dtype=torch.float32, device=device)

def get_probe_dir(probe):
    probe_weights = torch.tensor(probe.coef_, dtype=torch.float32, device=device).squeeze()
    return probe_weights / probe_weights.norm(dim=-1)


# calculate the ITI addition (sigma * theta) for one head
# uses either MMD or probe
def calc_truth_proj(activation, use_MMD=False, use_probe=False, truth_indices=None, probe=None):
    '''
    activation is (batch, d_m)
    '''
    if use_MMD: # use mass mean direction -- average difference between true and false classified prompts (only one head)
        assert truth_indices is not None
        truthful_dir = get_mass_mean_dir(activation, truth_indices)
    else: # probe -- just the coefficients of the probe
        assert use_probe
        assert probe is not None
        truthful_dir = get_probe_dir(probe)

    # print(f"Old truthful dir direc is {truthful_dir.shape}")
    truthful_dir /= truthful_dir.norm(dim=-1)
    # print(f"New truthful dir direc is {truthful_dir.shape}")
    act_std = get_act_std(activation, truthful_dir)
    
    return einops.einsum(act_std, truthful_dir, "d_m, d_m -> d_m")

def patch_activation_hook_fn(activations, hook: HookPoint, head, old_activations, use_MMD=True, use_probe=False, truth_indices=None, probe=None):
    """
    activations: (batch, n_heads, d_model)
    hook: HookPoint
    term_to_add: (*, d_model)

    A hook that is meant to act on the "z" (output) of a given head, and add the "term_to_add" on top of it. Only meant to work a certain head. Will broadcast.
    """
    # print(f"in hook fn, old act shape is {old_activations.shape}")
    term_to_add = calc_truth_proj(old_activations[:,head], use_MMD, use_probe, truth_indices, probe)
    # print(f"v shape is {term_to_add.shape}")
    # print(f"activations shape is {activations.shape}")
    activations[:,-1,head] += term_to_add

# Calculates new_activations for topk and adds temporary hooks
def patch_top_activations(model, probe_accuracies, old_activations, topk=20, alpha=20, use_MMD=False, use_probe=False, truth_indices=None, probes=None):
    '''
    probe_accuracies: (n_layers, n_heads)
    old_activations: (batch, n_layers, n_heads, d_model)

    if use_probe is True, probes should be list in shape (n_layers, n_heads) filled with probes

    Goes into every single activation, and then tells it to add the ITI
    '''

    # print(f"old activations shape is {old_activations.shape}")

    top_head_indices = torch.topk(einops.rearrange(probe_accuracies, "n_l n_h -> (n_l n_h)"), k=topk).indices # take top k indices
    top_head_bools = torch.zeros(size=(probe_accuracies.shape[0] * probe_accuracies.shape[1],)) # set all the ones that aren't top to 0

    top_head_bools[top_head_indices] = torch.ones_like(top_head_bools[top_head_indices]) # set all the ones that are top to 1
    top_head_bools = einops.rearrange(top_head_bools, "(n_l n_h) -> n_l n_h", n_l=model.cfg.n_layers) # rearrange back
    
    for layer in range(probe_accuracies.shape[0]):
        for head in range(probe_accuracies.shape[1]):
            if top_head_bools[layer, head] == 1:

                if use_probe:
                    patch_activation_with_head = partial(patch_activation_hook_fn, head = head, old_activations = old_activations[:, layer], use_MMD=False, use_probe=use_probe, truth_indices=None, probe=probes[layer][head])
                else:
                    patch_activation_with_head = partial(patch_activation_hook_fn, head = head, old_activations = old_activations[:, layer], use_MMD=use_MMD, use_probe=False, truth_indices=truth_indices, probe=None)
                model.add_hook(utils.get_act_name("result", layer), patch_activation_with_head)
