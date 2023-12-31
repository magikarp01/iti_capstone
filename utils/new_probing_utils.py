import os
from einops import repeat
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
from tqdm import tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.set_grad_enabled(False)
from sklearn.model_selection import train_test_split
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import pickle
from collections import defaultdict

from utils.dataset_utils import Abstract_Dataset
from utils.torch_hooks_utils import HookedModule
from utils.dataset_utils import TorchSample
from functools import partial
from jaxtyping import Float
from torch import Tensor

from typing import TypeVar
ModelActs = TypeVar("ModelActs")

from dataclasses import dataclass
import pickle
from abc import abstractclassmethod

@dataclass
class ModelActs:
    """
    A dataclass to record model activations ran on a user-defined dataset for later use. ModelActs has utilities to train probes on stored activations to elicit the truth.

    Acts should be generated or loaded by child classes. Then, split acts into train and test using set_train_test_split, and train probes on these acts using train_probes: these probes are stored in self.probes dictionary and accuracies are stored in self.probe_accs dictionary.

    Args:
        activations: nested dictionary, nested dictionary has index of component ((layer, head) for z) as key and stored activations as value, shape of (num_samples, componenent_dimension).
        probes: nested dictionary, LogisticRegression probe as value.
        probe_accs: nested dictionary, probe accuracy as value.

        labels: truth labels for training probes, either 0 or 1. Shape of (num_samples, ) corresponding to labels of same data indices as activations.
        indices_trains: indices of activations used to train probe. A subset of np.arange(num_samples).
        indices_tests: reserved activation/test indices (not used to train probe). A subset of np.arange(num_samples).
    """
    
    # def __init__(self, activations={}, probes={}, probe_accs={}, labels=None, indices_trains=None, indices_tests=None):
    def __init__(self):
        # each of these is a dictionary, keys are act type
        # self.activations = activations
        # self.probes = probes
        # self.probe_accs = probe_accs

        # self.labels: np.ndarray or torch.T = labels
        # self.indices_trains: np.array = indices_trains
        # self.indices_tests: np.array = indices_tests
        self.activations = {}
        self.probes = {}
        self.probe_accs = {}

        self.labels: np.ndarray or torch.T = None
        self.indices_trains: np.array = None
        self.indices_tests: np.array = None


    @abstractclassmethod
    def load_acts(self):
        """
        Load activations into self.activations dictionary and labels into self.labels. Should be implemented by child classes.
        """
        raise NotImplementedError


    def set_train_test_split(self, num_data, test_ratio = 0.2, train_ratio = None):
        """
        Given number of data points, sets train/test split of indices. test_ratio is proportion of data to be used for testing, train_ratio defaults to 1-test_ratio but can be set.
        Indices go from 0 to num_data - 1. 
        """
        act_indices = np.arange(num_data)
        
        if train_ratio is not None:
            assert test_ratio + train_ratio <= 1
        else:
            train_ratio = 1 - test_ratio

        self.indices_trains, self.indices_tests = train_test_split(act_indices, test_size=test_ratio, train_size=train_ratio)


    def _train_probe(self, act_type, probe_index, max_iter=1000):
        """
        Train a single logistic regression probe on input activations X_acts and labels (either 1 or 0 for truth or false). 
        Trains on train_indices of X_acts and labels, tests on test_indices.
        
        Args:
            act_type: type of activations to train probe on, "z" or "mlp_out" or etc.
            probe_index: index of probe to train, (layer, head) for z.

        Returns probe, accuracy
        """

        X_acts = self.activations[act_type][probe_index]

        X_train_head = X_acts[self.indices_trains] # integer array indexing
        y_train = self.labels[self.indices_trains]

        X_test_head = X_acts[self.indices_tests]
        y_test = self.labels[self.indices_tests]

        clf = LogisticRegression(max_iter=max_iter).fit(X_train_head, y_train)
        # clf = LogisticRegression(max_iter=max_iter).fit(self.activations[act_type][probe_index][self.indices_trains], self.labels[self.indices_trains])

        y_val_pred = clf.predict(X_test_head)
        # y_val_pred = clf.predict(self.activations[act_type][probe_index][self.indices_tests])
        acc = accuracy_score(y_test, y_val_pred)

        return clf, acc


    def train_probes(self, act_type, test_ratio=0.2, train_ratio=None, max_iter=1000):
        """
        Train probes on all provided activations of act_type in self.activations.
        If train test split is not already set, set it with given keywords.
        """
        if self.indices_tests is None and self.indices_trains is None:

            act_index = list(self.activations[act_type].keys())[0]
            num_acts = self.activations[act_type][act_index].shape[0]

            self.set_train_test_split(num_acts, test_ratio=test_ratio, train_ratio=train_ratio)
        
        if act_type not in self.probes:
            self.probes[act_type] = {}
            self.probe_accs[act_type] = {}

        for probe_index in tqdm(self.activations[act_type]):
            clf, acc = self._train_probe(act_type, probe_index, max_iter=max_iter)
            self.probes[act_type][probe_index] = clf
            self.probe_accs[act_type][probe_index] = acc


    def save_probes(self, id, filepath = "activations/"):
        """
        Save probes and probe accuracies to activations folder. Must be called after train_probes.
        """
        with open(f'{filepath}{id}_probes.pickle', 'wb') as handle:
            pickle.dump(self.probes, handle)

        with open(f'{filepath}{id}_probes_accs.pickle', 'wb') as handle:
            pickle.dump(self.probe_accs, handle)


    def show_top_z_probes(self, topk=50):
        """
        Utility to print the most accurate heads. 
        """
        assert "z" in self.probe_accs

        probe_accuracies = torch.tensor(einops.rearrange(self.probe_accs["z"], "(n_l n_h) -> n_l n_h", n_l=self.model.cfg.n_layers))
        top_head_indices = torch.topk(einops.rearrange(probe_accuracies, "n_l n_h -> (n_l n_h)"), k=topk).indices # take top k indices

        top_probe_heads = torch.zeros(size=(self.model.cfg.total_heads,))
        top_probe_heads[top_head_indices] = 1
        top_probe_heads = einops.rearrange(top_probe_heads, "(n_l n_h) -> n_l n_h", n_l=self.model.cfg.n_layers)
        for l in range(self.model.cfg.n_layers):
            for h in range(self.model.cfg.n_heads):
                if top_probe_heads[l, h] == 1:
                    print(f"{l}.{h}, ", end=" ")


    def get_probe_transfer_acc(self, act_type, probe_index, data_source: ModelActs):
        """
        Get transfer accuracy of probes trained on these model acts on another set of model acts, on new data. 
        data_source is another ModelActs object that should already have been trained on probes for act_type (with its own train test split).
        """
        
        data_acts = data_source.activations[act_type][probe_index]
        data_labels = data_source.labels # test indices from data_source

        probe = self.probes[act_type][probe_index]
        y_pred = probe.predict(data_acts)
        acc = accuracy_score(data_labels, y_pred)

        return acc



class SmallModelActs(ModelActs):
    """
    A class to handle small model activations ran on a user-defined dataset. Class has utilities to generate new activations based on a given model and dataset using TransformerLens (only compatible with 1 GPU, which is why model has to be small). 
    
    Generate and store all activations/probes for all heads at once. Not memory efficient, but all-in-one for small models.

    Initialize ModelActs class specifying which activation types to store.
    First, generate acts using gen_acts. Acts can be saved/loaded. Then, train probes on these acts using train_probes: these probes are stored in self.probes dictionary and accuracies are stored in self.probe_accs dictionary.
    """
    def __init__(self, model: HookedTransformer, dataset: Abstract_Dataset, seed = None, act_types = ["z"]):
        """
        dataset must have sample(sample_size) method returning indices of samples, sample_prompts, and sample_labels.
        act_types is a list of which activations to cache and operate on.
        """
        super().__init__()
        self.model = model
        # self.model.cfg.total_heads = self.model.cfg.n_heads * self.model.cfg.n_layers
        self.dataset = dataset
        
        # indices of data
        self.data_indices = None
        self.act_types=act_types

        if "result" in act_types:
            model.set_use_attn_result(True)
    
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def _tensor_to_dict(self, tensor, head_tensor=False):
        """
        Helper method to convert a tensor in shape (component_pos, ...) into a dict of (component_pos) -> entry. 

        If head_tensor is True, instead converts a tensor in shape (n_l, n_h, ...) into dict of (n_l, n_h) -> entry.
        """
        assert len(tensor.shape) >= 1
        if head_tensor:
            assert len(tensor.shape) >= 2
        
        entry_dict = {}
        
        if head_tensor:
            for i in range(tensor.shape[0]):
                for j in range(tensor.shape[1]):
                    entry_dict[(i, j)] = tensor[i, j]
        else:
            for i in range(tensor.shape[0]):
                entry_dict[i] = tensor[i]

        return entry_dict

    def gen_acts(self, N = 1000, store_acts = True, filepath = "activations/", id = None, indices=None, storage_device="cpu", verbose=False):
        """
        Automatically generates activations over N samples (returned in self.indices). If store_acts is True, then store in activations folder. Indices are indices of samples in dataset.
        
        Args:
            N: number of samples to generate activations over.
            store_acts: whether to store activations in activations folder.
            filepath: where to store activations.
            id: id of activations, name used for storing activations.
            indices: indices of samples of dataset to generate activations over.
            storage_device: device to store activations on.
        """        
        if indices is None:
            data_indices, all_prompts, all_labels = self.dataset.sample(N)
        else:
            data_indices = indices

        cached_acts = defaultdict(list)

        # names filter for efficiency, only cache in self.act_types
        names_filter = lambda name: any([name.endswith(act_type) for act_type in self.act_types])

        for i in tqdm(data_indices):
            original_logits, cache = self.model.run_with_cache(self.dataset.all_prompts[i].to(self.model.cfg.device), names_filter=names_filter) # only cache desired act types
            
            # store every act type in self.act_types
            for act_type in self.act_types:

                if act_type == "result":
                    # get last seq position
                    stack_acts = cache.stack_head_results(layer=-1, pos_slice=-1).squeeze().to(device=storage_device)
                    stack_acts = einops.rearrange(stack_acts, "(n_l n_h) d -> n_l n_h d", n_l = self.model.cfg.n_layers)
                
                elif act_type == "logits":
                    stack_acts = original_logits[:,-1].squeeze().to(device=storage_device) # logits of last token, shape (vocab_size,)

                else:
                    stack_acts = cache.stack_activation(act_type, layer = -1).to(device=storage_device)

                    if act_type == "attn_scores":
                        # shape of cache.stack_activation(act_type, layer = -1) is (n_l, 1, n_h, s, s)
                        stack_acts = stack_acts.squeeze().to(device=storage_device)

                    else:
                        # z or mlp or resid 
                        # shape (n_l, 1, seq_len, n_h, d_h) or (n_l, 1, seq_len, d_m)
                        stack_acts = stack_acts[:,0,-1].squeeze()

                if verbose:
                    print(f"{act_type=}, {stack_acts.shape=}")

                cached_acts[act_type].append(stack_acts)
        
        if verbose:
            print("Finished generating activations")

        for act_type in self.act_types:
            try:
                stacked_acts = einops.rearrange(torch.stack(cached_acts[act_type], dim=0), "n_acts ... d -> ... n_acts d")
                cached_acts[act_type] = stacked_acts
            except:
                print(f"Cannot stack activations, {act_type} not implemented yet")
                stacked_acts = cached_acts[act_type]   
                continue
        
        if verbose:
            print("Activations stacked")
        
        for act_type in self.act_types:
            if verbose:
                print(f"{act_type}, ")
            stacked_acts = cached_acts[act_type]
            if act_type == "result" or act_type == "z":
                act_dict = self._tensor_to_dict(stacked_acts, head_tensor=True)
            elif act_type == "logits":
                act_dict = {0: stacked_acts}
            elif act_type == "attn_scores":
                print("attn_scores not implemented yet")
                continue
            else:
                # mlp or resid
                assert len(stacked_acts.shape) == 3
                act_dict = self._tensor_to_dict(stacked_acts, head_tensor=False)

            self.activations[act_type] = act_dict
        
        if verbose:
            print("Finished formatting")

        self.data_indices = data_indices
        self.labels = np.array(all_labels)

        if store_acts:
            if id is None:
                id = np.random.randint(10000)
            torch.save(self.data_indices, f'{filepath}{id}_data_indices.pt')
            for act_type in self.act_types:
                with open(f'{filepath}{id}_{act_type}_acts.pt', 'wb') as f:
                    pickle.dump(self.activations[act_type], f)
            print(f"Stored at {id}")


    def load_acts(self, id, filepath = "activations/", load_probes=False):
        """
        Loads activations from activations folder. If id is None, then load the most recent activations. 
        If load_probes is True, load from saved probes.picle and all_heads_acc_np.npy files as well.
        """
        data_indices = torch.load(f'{filepath}{id}_data_indices.pt')

        for act_type in self.act_types:
            with open(f'{filepath}{id}_{act_type}_acts.pt', 'rb') as f:
                self.activations[act_type] = pickle.load(f)
        
        self.data_indices = data_indices

        if load_probes:
            print("load_probes deprecated for now, please change to False")
            with open(f'{filepath}{id}_probes.pickle', 'rb') as handle:
                self.probes = pickle.load(handle)
            self.all_head_accs_np = np.load(f'{filepath}{id}_all_head_accs_np.npy')


    def control_for_iti(self, cache_interventions):
        """
        Subtracts the actual iti intervention vectors from the cached activations: this removes the direct effect of ITI and helps us see the downstream effects.
        
        Args:
            cache_interventions: tensor of shape (n_l, n_h, d_h) or (n_l, d_m)
        """
        # self.stored_acts["z"] -= einops.rearrange(cache_interventions, "n_l n_h d_h -> (n_l n_h) d_h")
        intervention_dict = self._tensor_to_dict(cache_interventions, head_tensor=True)

        for probe_index in self.activations["z"]:
            self.activations["z"][probe_index] -= intervention_dict[probe_index]


class ModelActsLargeSimple(ModelActs):
    """
    A class that implements functionality of ModelActs, but optimized for large models. This class doesn't generate the activations itself: instead, it loads the activations of specific components and probes only those.

    * presumes that we already have formatted activations
    """

    def load_acts(self, file_prefix, n_layers, n_heads, labels,component_indices=None, exclude_points=None):
        """
        Load z activations and labels from formatted_acts directory. 

        Args:
            file_prefix: prefix of the file name including directories, e.g. "data/large_run_1/activations/formatted/large_run_1_honest"
            n_layers and n_heads: number of layers and heads in the model
            labels: labels of the data in numpy array (for now, loaded externally from huggingface)
            component_indices: which heads to store and load. If none, default to loading and storing all heads. Should be a boolean array of shape (n_l, n_h)
            exclude_points: datapoints (indices) to exclude for any reason
        """
        if component_indices is None:
            component_indices = np.full(shape=(n_layers, n_heads), fill_value=True)
        
        self.labels = labels

        if "z" not in self.activations:
            self.activations["z"] = {}
        for layer in tqdm(range(n_layers)):
            for head in range(n_heads):
                if component_indices[layer, head]:
                    X_acts = torch.load(f"{file_prefix}_l{layer}_h{head}.pt")
                    mask = torch.any(X_acts != 0, dim=1)
                    if exclude_points is not None:
                        for point in exclude_points:
                            mask[point] = False
                    X_acts = X_acts[mask]
                    
                    self.activations["z"][(layer, head)] = X_acts.numpy()

