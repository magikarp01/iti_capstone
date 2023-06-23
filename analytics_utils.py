import numpy as np
import torch
import plotly.express as px

def plot_probe_accuracies(model_acts):
    accs_sorted = -np.sort(-model_acts.all_head_accs_np.reshape(model_acts.model.cfg.n_layers, model_acts.model.cfg.n_heads), axis = 1)
    return px.imshow(accs_sorted, labels = {"x" : "Heads (sorted)", "y": "Layers"},title = "Probe Accuracies", color_continuous_midpoint = 0.5, color_continuous_scale="YlGnBu", origin = "lower")

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
