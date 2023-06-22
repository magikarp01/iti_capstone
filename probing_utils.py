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
import tqdm.notebook as tqdm
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


# from iti import patch_top_activations

"""
A class to handle model activations ran on a user-defined dataset. Class has utilities to generate new
activations based on a given model and dataset, and to store those activations for later use. ModelActs also
has utilities to train probes on the activations of every head.
"""
class ModelActs():
    def __init__(self, model: HookedTransformer, dataset, seed = 0):
        """
        dataset must have sample(sample_size) method returning indices of samples, sample_prompts, and sample_labels.
        """
        self.model = model
        # self.model.cfg.total_heads = self.model.cfg.n_heads * self.model.cfg.n_layers
        self.dataset = dataset
        
        self.attn_head_acts = None
        self.indices = None
    
        np.random.seed(seed)
        torch.manual_seed(seed)

    def get_acts(self, N = 1000, store_acts = True, filepath = "activations/", id = None):
        
        attn_head_acts = []
        indices, all_prompts, all_labels = self.dataset.sample(N)
        
        for i in tqdm(indices):
                original_logits, cache = self.model.run_with_cache(self.dataset.all_prompts[i].to(self.model.device))
                attn_head_acts.append(cache.stack_head_results(layer = -1, pos_slice = -1).squeeze(1))
        
        self.attn_head_acts = torch.stack(attn_head_acts).reshape(-1, self.model.cfg.total_heads, self.model.cfg.d_model)
        self.indices = indices
        
        if store_acts:
            if id is None:
                id = np.random.randint(10000)
            torch.save(self.indices, f'{filepath}{id}_indices.pt')
            torch.save(self.attn_head_acts, f'{filepath}{id}_attn_head_acts.pt')
            print(f"Stored at {id}")

        return self.indices, self.attn_head_acts

    def load_acts(self, id, filepath = "activations/"):
        indices = torch.load(f'{filepath}{id}_indices.pt')
        attn_head_acts = torch.load(f'{filepath}{id}_attn_head_acts.pt')
        
        self.attn_head_acts = attn_head_acts
        self.indices = indices
        self.probes = None
        self.all_head_accs_np = None
        
        return indices, attn_head_acts

    def get_train_test_split(self, test_ratio = 0.2, N = None):
        attn_head_acts_list = [self.attn_head_acts[i] for i in range(self.attn_head_acts.shape[0])]
        
        indices = self.indices
        
        if N is not None:
            attn_heads_acts_list = attn_heads_acts_list[:N]
            indices = indices[:N]
        
        # print(self.attn_head_acts.shape)
        # print(len(self.dataset.all_labels))
        # print(len(indices))
        # print(np.array(self.dataset.all_labels)[indices])
        # print(len(np.array(self.dataset.all_labels)[indices]))
        
        X_train, X_test, y_train, y_test = train_test_split(attn_head_acts_list, np.array(self.dataset.all_labels)[indices], test_size=test_ratio)
        
        X_train = torch.stack(X_train, axis = 0)
        X_test = torch.stack(X_test, axis = 0)  
        
        y_train = torch.from_numpy(np.array(y_train, dtype = np.float32))
        y_test = torch.from_numpy(np.array(y_test, dtype = np.float32))
        y_train = repeat(y_train, 'b -> b num_attn_heads', num_attn_heads=self.model.cfg.total_heads)
        y_test = repeat(y_test, 'b -> b num_attn_heads', num_attn_heads=self.model.cfg.total_heads)

        return X_train, X_test, y_train, y_test

    def train_probes(self):
        """
        Train linear probes on every head's activations. Must be called after either get_acts or load_acts.
        Probes are stored in self.probes, and accuracies are stored in self.all_head_accs_np (also returned).
        """
        X_train, X_test, y_train, y_test = self.get_train_test_split()
        print(f"{X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}")
        
        all_head_accs = []
        probes = []
        
        for i in tqdm(range(self.model.cfg.total_heads)):
                X_train_head = X_train[:,i,:]
                X_test_head = X_test[:,i,:]

                clf = LogisticRegression(max_iter=1000).fit(X_train_head.detach().numpy(), y_train[:, 0].detach().numpy())
                y_pred = clf.predict(X_train_head)
                
                y_val_pred = clf.predict(X_test_head.detach().numpy())
                all_head_accs.append(accuracy_score(y_test[:, 0].numpy(), y_val_pred))
                
                probes.append(clf)

        self.probes = probes
        
        self.all_head_accs_np = np.array(all_head_accs)
        
        return self.all_head_accs_np