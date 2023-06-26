import numpy as np
import torch
import plotly.express as px
import einops

def plot_probe_accuracies(model_acts, sorted = False, other_head_accs=None, title = "Probe Accuracies"):
    """
    Takes a model_acts (ModelActs) object by default. If other_head_accs is not None, then it must be a tensor of head accs, and other_head_accs is plotted.
    """

    if other_head_accs is not None:
        all_head_accs_np = other_head_accs
    else:
        all_head_accs_np = model_acts.all_head_accs_np

    if sorted:
        accs_sorted = -np.sort(-all_head_accs_np.reshape(model_acts.model.cfg.n_layers, model_acts.model.cfg.n_heads), axis = 1)
    else:
        accs_sorted = all_head_accs_np.reshape(model_acts.model.cfg.n_layers, model_acts.model.cfg.n_heads)
    return px.imshow(accs_sorted, labels = {"x" : "Heads (sorted)", "y": "Layers"}, title = title, color_continuous_midpoint = 0.5, color_continuous_scale="YlGnBu", origin = "lower")

def plot_norm_diffs(model_acts_iti, model_acts, div=True):
    """
    div = True means divide by original act norms
    """

    norm_diffs = torch.norm((model_acts_iti.attn_head_acts - model_acts.attn_head_acts), dim = 2).mean(0)
    if div:
        norm_diffs /= torch.norm(model_acts.attn_head_acts, dim = 2).mean(0)
    
    norm_diffs = norm_diffs.numpy().reshape(model_acts.model.cfg.n_layers, model_acts.model.cfg.n_heads)

    return px.imshow(norm_diffs, labels = {"x" : "Heads", "y": "Layers"},title = "Norm Differences (divided by original norm) of ITI and Normal Head Activations", color_continuous_midpoint = 0, color_continuous_scale="RdBu", origin = "lower")

def plot_cosine_sims(model_acts_iti, model_acts):
    act_sims = torch.nn.functional.cosine_similarity(model_acts_iti.attn_head_acts, model_acts.attn_head_acts, dim=2).mean(0)
    act_sims = act_sims.numpy().reshape(model_acts_iti.model.cfg.n_layers, model_acts_iti.model.cfg.n_heads)

    # act_sims[44, 23] = act_sims[45, 17] = 1
    return px.imshow(act_sims, labels = {"x" : "Heads", "y": "Layers"},title = "Cosine Similarities of of ITI and Normal Head Activations", color_continuous_midpoint = 1, color_continuous_scale="RdBu", origin = "lower")

def plot_downstream_diffs(model_acts_iti, model_acts, cache_interventions, div=True):
    """
    div = True means divide by original act norms
    cache_interventions is pytorch tensor, shape n_l, n_h, d_h
    """

    act_difference = model_acts_iti.attn_head_acts - model_acts.attn_head_acts # (b, (n_l n_h), d_h)
    act_difference -= einops.rearrange(cache_interventions, "n_l n_h d_h -> (n_l n_h) d_h") 

    norm_diffs = torch.norm((act_difference), dim = 2).mean(0)
    if div:
        norm_diffs /= torch.norm(model_acts.attn_head_acts, dim = 2).mean(0)
    
    norm_diffs = norm_diffs.numpy().reshape(model_acts.model.cfg.n_layers, model_acts.model.cfg.n_heads)

    return px.imshow(norm_diffs, labels = {"x" : "Heads", "y": "Layers"},title = "Norm Differences (divided by original norm) of ITI and Normal Head Activations", color_continuous_midpoint = 0, color_continuous_scale="RdBu", origin = "lower")

