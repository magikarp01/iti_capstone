import numpy as np
import torch
import plotly.express as px
import einops
import csv

def plot_probe_accuracies(model_acts, sorted = False, other_head_accs=None, title = "Probe Accuracies"):
    """
    Plots z probe accuracies.
    Takes a model_acts (ModelActs) object by default. If other_head_accs is not None, then it must be a tensor of head accs, and other_head_accs is plotted.
=    """

    if other_head_accs is not None:
        all_head_accs_np = other_head_accs
    else:
        all_head_accs_np = model_acts.probe_accs["z"]

    if sorted:
        accs_sorted = -np.sort(-all_head_accs_np.reshape(model_acts.model.cfg.n_layers, model_acts.model.cfg.n_heads), axis = 1)
    else:
        accs_sorted = all_head_accs_np.reshape(model_acts.model.cfg.n_layers, model_acts.model.cfg.n_heads)
    return px.imshow(accs_sorted, labels = {"x" : "Heads (sorted)", "y": "Layers"}, title = title, color_continuous_midpoint = 0.5, color_continuous_scale="YlGnBu", origin = "lower")

def plot_resid_probe_accuracies(acc_dict, n_layers, title = "Probe Accuracies", graph_type="line"):
    """
    Plot a dotted line graph with n_layer points, where each point is the accuracy of the residual probe at that layer.
    """
    accs = np.ones(shape=(n_layers)) * -1
    for layer in range(n_layers):
        accs[layer] = acc_dict[layer]
    
    if graph_type == "line":
        fig = px.line(accs, labels = {"x" : "Layers", "y": "Accuracy"}, title = title)
        fig.update_traces(line=dict(dash='dot'), marker=dict(size=3, color='blue')) 
        fig.update_layout(yaxis_title="Accuracy", showlegend=False)
    elif graph_type == "square":
        fig = px.imshow(np.expand_dims(accs, axis=1), labels = {"y": "Layers"}, title = title, color_continuous_midpoint = 0.5, color_continuous_scale="YlGnBu", origin = "lower")
        fig.update_xaxes(showticklabels=False)

    # color_continuous_midpoint = 0.5, color_continuous_scale="YlGnBu", origin = "lower"
    return fig

def plot_z_probe_accuracies(acc_dict, n_layers, n_heads, sorted = False, title = "Probe Accuracies"):
    """
    Plot z probe accuracies given an acc dict, with keys (layer, head) and value accuracy.
    """
    head_accs = np.ones(shape=(n_layers, n_heads)) * -1
    for layer in range(n_layers):
        for head in range(n_heads):
            if (layer, head) in acc_dict:
                head_accs[layer, head] = acc_dict[(layer, head)]
    
    if sorted:
        head_accs = -np.sort(-head_accs, axis = 1)

    return px.imshow(head_accs, labels = {"x" : "Heads (sorted)", "y": "Layers"}, title = title, color_continuous_midpoint = 0.5, color_continuous_scale="YlGnBu", origin = "lower")


def plot_norm_diffs(model_acts_iti, model_acts, div=True):
    """
    Plots the norm diffs across head z activations
    div = True means divide by original act norms
    """

    iti_acts = einops.rearrange(model_acts_iti.stored_acts["z"], "b n_l n_h d_p -> b (n_l n_h) d_p")
    orig_acts = einops.rearrange(model_acts.stored_acts["z"], "b n_l n_h d_p -> b (n_l n_h) d_p")

    norm_diffs = torch.norm((iti_acts - orig_acts), dim = 2).mean(0)
    if div:
        norm_diffs /= torch.norm(orig_acts, dim = 2).mean(0)
    
    norm_diffs = norm_diffs.numpy().reshape(model_acts.model.cfg.n_layers, model_acts.model.cfg.n_heads)

    return px.imshow(norm_diffs, labels = {"x" : "Heads", "y": "Layers"},title = "Norm Differences (divided by original norm) of ITI and Normal Head Activations", color_continuous_midpoint = 0, color_continuous_scale="RdBu", origin = "lower")

def plot_cosine_sims(model_acts_iti, model_acts):
    iti_acts = einops.rearrange(model_acts_iti.stored_acts["z"], "b n_l n_h d_p -> b (n_l n_h) d_p")
    orig_acts = einops.rearrange(model_acts.stored_acts["z"], "b n_l n_h d_p -> b (n_l n_h) d_p")

    act_sims = torch.nn.functional.cosine_similarity(iti_acts, orig_acts, dim=2).mean(0)
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

def get_inference_accuracy(filename, threshold=0):
    num_correct = 0
    num_total = 0
    acc = 0
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            if idx>0:
                p_true = float(row[1])
                p_false = float(row[2])
                if p_true > threshold or p_false > threshold:
                    label = int(float(row[3]))
                    
                    pred = p_true > p_false
                    correct = (pred == label) #bool

                    num_correct += correct
                    num_total += 1
    if num_total > 0:
        acc = num_correct / num_total
    return acc, num_total