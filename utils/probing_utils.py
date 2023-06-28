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
from torch.utils.data import DataLoader

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

from typing import TypeVar
ModelActs = TypeVar("ModelActs")


class ModelActs:
    """
    A class to handle model activations ran on a user-defined dataset. Class has utilities to generate new activations based on a given model and dataset, and to store those activations for later use. ModelActs also has utilities to train probes on the activations of every head.

    Initialize ModelActs class specifying which activation types to store.
    First, generate acts using gen_acts. Acts can be saved/loaded. Then, train probes on these acts using train_probes: these probes are stored in self.probes dictionary and accuracies are stored in self.probe_accs dictionary.
    """
    def __init__(self, model: HookedTransformer, dataset, seed = None, act_types = ["z"]):
        """
        dataset must have sample(sample_size) method returning indices of samples, sample_prompts, and sample_labels.
        act_types is a list of which activations to cache and operate on.
        """
        self.model = model
        # self.model.cfg.total_heads = self.model.cfg.n_heads * self.model.cfg.n_layers
        self.dataset = dataset
        
        self.attn_head_acts = None
        self.indices = None
        
        self.act_types=act_types
    
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.probes = {}
        self.probe_accs = {}
        self.X_tests = {}
        self.y_tests = {}
        self.indices_trains = {}
        self.indices_tests = {}

    """
    Automatically generates activations over N samples (returned in self.indices). If store_acts is True, then store in activations folder. Indices are indices of samples in dataset.
    Refactored so that activations are not reshaped by default, and will be reshaped at some other time.
    """
    def gen_acts(self, N = 1000, store_acts = True, filepath = "activations/", id = None, indices=None, storage_device="cpu"):
        
        if indices is None:
            indices, all_prompts, all_labels = self.dataset.sample(N)
        
        cached_acts = defaultdict(list)

        # names filter for efficiency, only cache in self.act_types
        names_filter = lambda name: any([name.endswith(act_type) for act_type in self.act_types])

        for i in tqdm(indices):
            original_logits, cache = self.model.run_with_cache(self.dataset.all_prompts[i].to(self.model.cfg.device), names_filter=names_filter) # only cache z
            
            # store every act type in self.act_types
            for act_type in self.act_types:

                if act_type == "result":
                    # get last seq position
                    stored_acts = cache.stack_head_results(layer=-1, pos_slice=-1).squeeze().to(device=storage_device)
                
                elif act_type == "logits":
                    stored_acts = original_logits[:,-1].to(device=storage_device) # logits of last token

                else:
                    stored_acts = cache.stack_activation(act_type, layer = -1)[:,0,-1].squeeze().to(device=storage_device)
                cached_acts[act_type].append(stored_acts)
        
        # convert lists of tensors into tensors
        stored_acts = {act_type: torch.stack(cached_acts[act_type]) for act_type in self.act_types} 

        self.stored_acts = stored_acts
        self.indices = indices
        
        if store_acts:
            if id is None:
                id = np.random.randint(10000)
            torch.save(self.indices, f'{filepath}{id}_indices.pt')
            for act_type in self.act_types:
                torch.save(self.stored_acts[act_type], f'{filepath}{id}_{act_type}_acts.pt')
            print(f"Stored at {id}")

        # return self.indices, self.attn_head_acts

    """
    Loads activations from activations folder. If id is None, then load the most recent activations. 
    If load_probes is True, load from saved probes.picle and all_heads_acc_np.npy files as well.
    """
    def load_acts(self, id, filepath = "activations/", load_probes=False):
        indices = torch.load(f'{filepath}{id}_indices.pt')

        self.stored_acts = {}
        for act_type in self.act_types:
            self.stored_acts[act_type] = torch.load(f'{filepath}{id}_{act_type}_acts.pt')
        
        self.indices = indices

        if load_probes:
            print("load_probes deprecated for now, please change to False")
            with open(f'{filepath}{id}_probes.pickle', 'rb') as handle:
                self.probes = pickle.load(handle)
            self.all_head_accs_np = np.load(f'{filepath}{id}_all_head_accs_np.npy')

        else:
            self.probes = {}
            # self.all_head_accs_np = None
            self.probe_accs = {}

        # return indices, attn_head_acts

    def control_for_iti(self, cache_interventions):
        """
        Deprecated for now.
        Subtracts the actual iti intervention vectors from the cached activations: this removes the direct
        effect of ITI and helps us see the downstream effects.
        """
        # self.stored_acts["z"] -= einops.rearrange(cache_interventions, "n_l n_h d_h -> (n_l n_h) d_h")
        self.stored_acts["z"] -= cache_interventions

    def get_train_test_split(self, X_acts, test_ratio = 0.2, N = None):
        """
        Given X_acts, a Pytorch tensor of shape (num_samples, num_probes, d_probe), and test ratio, split acts and labels into train and test sets.
        """
        X_acts_list = [X_acts[i] for i in range(X_acts.shape[0])]
        
        indices = self.indices
        
        if N is not None:
            X_acts_list = X_acts_list[:N]
            indices = indices[:N]
        
        # print(self.attn_head_acts.shape)
        # print(len(self.dataset.all_labels))
        # print(len(indices))
        # print(np.array(self.dataset.all_labels)[indices])
        # print(len(np.array(self.dataset.all_labels)[indices]))

        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_acts_list, np.array(self.dataset.all_labels)[indices], indices, test_size=test_ratio)
        
        X_train = torch.stack(X_train, axis = 0)
        X_test = torch.stack(X_test, axis = 0)
        
        y_train = torch.from_numpy(np.array(y_train, dtype = np.float32))
        y_test = torch.from_numpy(np.array(y_test, dtype = np.float32))
        y_train = repeat(y_train, 'b -> b num_probes', num_probes=X_acts.shape[1])
        y_test = repeat(y_test, 'b -> b num_probes', num_probes=X_acts.shape[1])

        return X_train, X_test, y_train, y_test, indices_train, indices_test


    def _train_probes(self, num_probes, X_train, X_test, y_train, y_test, max_iter=1000):
        """
        Helper function that all train_x_probes will call to train num_probes probes, after formatting X and y properly.
        Flatten X so that X_train.shape = (num_examples, num_probes, d_probe)
        y_train.shape = (num_examples, num_probes)
        Returns list of probes and np array of probe accuracies
        """
        all_head_accs = []
        probes = []
        
        for i in tqdm(range(num_probes)):
            X_train_head = X_train[:,i,:]
            X_test_head = X_test[:,i,:]

            clf = LogisticRegression(max_iter=max_iter).fit(X_train_head.detach().numpy(), y_train[:, 0].detach().numpy()) # for every probe i, y_train[:, i] is the same
            y_pred = clf.predict(X_train_head)
            
            y_val_pred = clf.predict(X_test_head.detach().numpy())
            all_head_accs.append(accuracy_score(y_test[:, 0].numpy(), y_val_pred))
            
            probes.append(clf)

        return probes, np.array(all_head_accs)


    '''
    def train_z_probes(self, max_iter=1000):
        """
        Train linear probes on every head's activations. Must be called after either gen_acts or load_acts.
        Probes are stored in self.probes, and accuracies are stored in self.all_head_accs_np (also returned).
        Also store last X_test and y_test used for probes
        """
        formatted_z_acts = einops.rearrange(self.stored_acts["z"], "b n_l n_h d_h -> b (n_l n_h) d_h")
        X_train, X_test, y_train, y_test, indices_train, indices_test = self.get_train_test_split(formatted_z_acts)
        # print(f"{X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}")

        probes, probe_accs = self._train_probes(formatted_z_acts.shape[1], X_train, X_test, y_train, y_test, max_iter=max_iter)

        self.X_tests["z"] = X_test
        self.y_tests["z"] = y_test
        self.indices_tests["z"] = indices_test
        
        self.probes["z"] = probes
        self.probe_accs["z"] = probe_accs


    def train_mlp_out_probes(self, max_iter=1000):
        X_train, X_test, y_train, y_test, indices_train, indices_test = self.get_train_test_split(self.stored_acts["mlp_out"])
        # print(f"{X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}")

        probes, probe_accs = self._train_probes(self.stored_acts["mlp_out"].shape[1], X_train, X_test, y_train, y_test, max_iter=max_iter)

        self.X_tests["mlp_out"] = X_test
        self.y_tests["mlp_out"] = y_test
        self.indices_tests["z"] = indices_test
        
        self.probes["mlp_out"] = probes
        self.probe_accs["mlp_out"] = probe_accs
    '''

    def train_probes(self, act_type, max_iter=1000):
        """
        Train arbitrary probes on any act type's activations (must've been included in self.act_types).
        self.stored_acts[act_type] should be shape (num_examples, ..., d_probe), will be flattened in the middle
        """
        assert act_type in self.act_types
        formatted_acts = torch.flatten(self.stored_acts[act_type], start_dim=1, end_dim=-2)

        assert len(formatted_acts.shape) == 3

        X_train, X_test, y_train, y_test, indices_train, indices_test = self.get_train_test_split(formatted_acts)
        print(f"{X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}")

        probes, probe_accs = self._train_probes(formatted_acts.shape[1], X_train, X_test, y_train, y_test, max_iter=max_iter)

        self.X_tests[act_type] = X_test
        self.y_tests[act_type] = y_test
        self.indices_trains[act_type] = indices_train
        self.indices_tests[act_type] = indices_test

        self.probes[act_type] = probes
        self.probe_accs[act_type] = probe_accs


    def save_probes(self, id, filepath = "activations/"):
        """
        Don't use, out of date since all_head_accs_np doesn't exist anymore.
        Save probes and probe accuracies to activations folder. Must be called after train_probes.
        """
        with open(f'{filepath}{id}_probes.pickle', 'wb') as handle:
            pickle.dump(self.probes, handle, protocol=pickle.HIGHEST_PROTOCOL)

        np.save(f'{filepath}{id}_all_head_accs_np.npy', self.all_head_accs_np)

    
    def get_transfer_acc(self, act_type, data_source: ModelActs):
        """
        Get transfer accuracy of probes trained on this dataset on another dataset. 
        data_source is another ModelActs object that should already have been trained on probes.
        """
        # data_labels = np.array(data_source.dataset.all_labels)[data_source.indices]
        data_labels = data_source.y_tests[act_type][:,0].numpy()

        accs = []

        for i, clf in tqdm(enumerate(self.probes[act_type])):
            # acts = data_source.attn_head_acts[:, i, :]
            acts = data_source.X_tests[act_type][:, i, :]
            y_pred = clf.predict(acts)
            
            accs.append(accuracy_score(data_labels, y_pred))
        
        return np.array(accs)

    def show_top_z_probes(self, topk=50):
        """
        Utility to print the most accurate heads. Out of date with probe_generalization merge.
        """
        probe_accuracies = torch.tensor(einops.rearrange(self.probe_accs["z"], "(n_l n_h) -> n_l n_h", n_l=self.model.cfg.n_layers))
        top_head_indices = torch.topk(einops.rearrange(probe_accuracies, "n_l n_h -> (n_l n_h)"), k=topk).indices # take top k indices

        top_probe_heads = torch.zeros(size=(self.model.cfg.total_heads,))
        top_probe_heads[top_head_indices] = 1
        top_probe_heads = einops.rearrange(top_probe_heads, "(n_l n_h) -> n_l n_h", n_l=self.model.cfg.n_layers)
        for l in range(self.model.cfg.n_layers):
            for h in range(self.model.cfg.n_heads):
                if top_probe_heads[l, h] == 1:
                    print(f"{l}.{h}, ", end=" ")