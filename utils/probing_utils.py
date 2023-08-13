from einops import repeat
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import gc
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

# ALWAYS export PYTORCH_NO_CUDA_MEMORY_CACHING=1 or you'll get an OOM on A100


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
        self.device = self.model.cfg.device # still need to replace all instances of this

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

#%% CCS modifications begin here

    def get_acts_pairs(self, N=1000, store_acts = True, filepath = "activations/", id = None, storage_device="cpu"):
        """
        Uses the CCS_Dataset class to generate many prompt activations, and places them on the storage_device = "cpu" by default.
        """

        cached_acts_yes = defaultdict(list)
        cached_acts_no = defaultdict(list)

        prompt_no, prompt_yes, y, used_idxs = self.dataset.sample_pair(N)

        # names filter for efficiency, only cache in self.act_types
        names_filter = lambda name: any([name.endswith(act_type) for act_type in self.act_types])

        for prompts, cached_acts in [(prompt_yes, cached_acts_yes), (prompt_no, cached_acts_no)]:
            for prompt in tqdm(prompts):
                
                original_logits, cache = self.model.run_with_cache(prompt, names_filter=names_filter)

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
        stored_acts_yes = {act_type: torch.stack(cached_acts_yes[act_type]) for act_type in self.act_types} 
        stored_acts_no = {act_type: torch.stack(cached_acts_no[act_type]) for act_type in self.act_types} 

        # CCS class uses these. All tensors.
        self.stored_acts_pairs = (stored_acts_yes, stored_acts_no, torch.Tensor(y), torch.Tensor(used_idxs))

        # if store_acts:
        #     if id is None:
        #         id = np.random.randint(10000)
        #     for act_type in self.act_types:
        #         torch.save(self.stored_acts[act_type], f'{filepath}{id}_{act_type}_acts.pt')
        #     print(f"Stored at {id}")

        return self.stored_acts_pairs

    def CCS_train(self, n_epochs, batch_size, act_type = "z"):
        """
        Runs arbitrary CCS probes on any act type's activations (must've been included in self.act_types).
        self.stored_acts_pairs for an act_type should be shape (num_examples, ..., d_probe), will be flattened in the middle.
        """
        stored_acts_yes, stored_acts_no, y, used_idxs = self.stored_acts_pairs
        stored_acts_yes = stored_acts_yes[act_type]
        stored_acts_no = stored_acts_no[act_type]

        # Flatten & get dimensions
        stored_acts_yes = torch.flatten(stored_acts_yes, start_dim=1, end_dim=-2)
        stored_acts_no = torch.flatten(stored_acts_no, start_dim=1, end_dim=-2)
        assert stored_acts_yes.shape == stored_acts_no.shape
        assert len(stored_acts_yes.shape) == 3
        num_samples = stored_acts_yes.shape[0]
        num_probes = stored_acts_yes.shape[1]
        probe_dim = stored_acts_yes.shape[2]

        # Normalize (if batch dimension is greater than 1)
        if num_samples > 1:
            stored_acts_yes = (stored_acts_yes - stored_acts_yes.mean(axis=0, keepdims=True)) / stored_acts_yes.std(axis=0, keepdims=True)
            stored_acts_no = (stored_acts_no - stored_acts_no.mean(axis=0, keepdims=True)) / stored_acts_no.std(axis=0, keepdims=True)

        # Train-test split
        indices = torch.randperm(num_samples) # random indices
        train_size = int(num_samples * 0.8)
        test_size = num_samples - train_size
        train_indices = indices[:train_size].to(torch.int64)
        test_indices = indices[train_size:].to(torch.int64)

        X_train_yes = stored_acts_yes[train_indices]
        X_train_no = stored_acts_no[train_indices]
        X_test_yes = stored_acts_yes[test_indices]  
        X_test_no = stored_acts_no[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        print(f"X_train_yes shape: {X_train_yes.shape}")
        print(f"X_train_no shape: {X_train_no.shape}")
        print(f"X_test_yes shape: {X_test_yes.shape}")
        print(f"X_test_no shape: {X_test_no.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

        # Train
        probe, cluster_label, accuracy = self._CCS_train(num_probes, X_train_yes, X_train_no, X_test_yes, X_test_no, y_train, y_test, n_epochs, batch_size)

        # Store
        self.CCS = {}
        self.CCS_cluster_label = {}
        self.CCS_accuracy = {}
        
        self.CCS[act_type] = probe
        self.CCS_cluster_label[act_type] = cluster_label
        self.CCS_accuracy[act_type] = accuracy

    def _CCS_train(self, num_probes, X_train_yes, X_train_no, X_test_yes, X_test_no, y_train, y_test, n_epochs, batch_size):

        # Shape check
        total_batch_size = X_train_yes.shape[0] # for train
        assert X_train_yes.shape[1] == num_probes
        probe_dim = X_train_yes.shape[2]

        print(f"Training {num_probes} probes of dimension {probe_dim} for {n_epochs} epochs with batch size {batch_size}.")
        
        probe = []
        cluster_label = []
        accuracy = []

        for i in tqdm(range(num_probes)):

            # Initialize probe, optimizer
            p0 = nn.Sequential(nn.Linear(probe_dim, 1), nn.Sigmoid()).to(self.model.cfg.device) # have to use pytorch for probe because custom loss function
            optimizer = torch.optim.AdamW(p0.parameters())

            for epoch in range(n_epochs):

                # get batch
                batch_indices = torch.randperm(total_batch_size)[:batch_size] # subset of batch this time
                acts_yes = X_train_yes[:,i,:].to(self.model.cfg.device)[batch_indices]
                acts_no = X_train_no[:,i,:].to(self.model.cfg.device)[batch_indices]

                differing_elements = torch.ne(acts_yes, acts_no).sum().item()
                print(f"There are {differing_elements} differing numbers between the two tensors.")

                optimizer.zero_grad()
                torch.set_grad_enabled(True) # tl sets grad enabled = False, so you have to do this every iteration loop
                p0_out, p1_out = p0(acts_yes), p0(acts_no)
                # get the corresponding loss
                informative_loss = (torch.min(p0_out, p1_out)**2).mean(0)
                consistent_loss = ((p0_out - (1-p1_out))**2).mean(0)
                loss = informative_loss + consistent_loss
                # update the parameters
                loss.backward()
                optimizer.step()
                torch.set_grad_enabled(False) # do this else huge memory leak from transformerlens

                # print probe parameters
                # print(f"Epoch {epoch}: {p0[0].weight.data}, {p0[0].bias.data}")

                # just in case, i can't be bothered w memory
                # del acts_yes
                # del acts_no
                # torch.cuda.empty_cache() 

            print("*** CLUSTER LABEL")
            print(f"X_train_yes, x_train_no, y_Train shape: {X_train_yes[:,i,:].shape}, {X_train_no[:,i,:].shape}, {y_train.shape}")
            cluster_label.append(self._CCS_label_clusters(X_train_yes[:,i,:], X_train_no[:,i,:], y_train, p0))
            print("*** ACCURACY")
            print(f"X_test_yes, x_test_no, y_test shape: {X_test_yes[:,i,:].shape}, {X_test_no[:,i,:].shape}, {y_test.shape}")
            accuracy.append(self._CCS_inference(X_test_yes[:,i,:], X_test_no[:,i,:], y_test, p0, cluster_label[i]))
            probe.append(p0)

        return probe, cluster_label, accuracy

    def _CCS_label_clusters(self, acts_yes, acts_no, labels, probe):
        """
        Gives the correct "direction" for the CCS probes given some data.

        False means that the probe is already in the "correct direction" where a probe returns the probability of the true label, while True means that you should flip the probe direction because the probe returns the probability of the false label.
        """
        acc = self._CCS_inference(acts_yes, acts_no, labels, probe, cluster_label = None)
        if acc > 0.5:
            return False # don't turn
        else:
            return True # do turn
    
    def _CCS_inference(self, acts_yes, acts_no, labels, probe, cluster_label = None):
        """
        Returns CCS predictions (probabilities) as well as accuracies (when you threshold at 0.5).
        Takes in input (batch_size, d_probe) for acts_yes and acts_no.
        """
        assert acts_yes.shape == acts_no.shape
        print(f"Labels: {labels}")
        with torch.no_grad():
            p0_out, p1_out = probe(acts_yes.to(self.model.cfg.device)), probe(acts_no.to(self.model.cfg.device))
        avg_confidence = 0.5*(p0_out + (1-p1_out))
        print(f"Average confidence: {avg_confidence.shape}") # torch.Size([2, 1])
        print(f"Average confidence: {avg_confidence}")
        predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int)[:, 0]
        print(f"Predictions: {predictions.shape}")
        print(f"Predictions: {predictions}")
        acc = (predictions == labels.numpy()).mean()
        print(f"Accuracy: {acc.shape}")
        print(f"Accuracy: {acc}")
        if cluster_label is not None and cluster_label == True: # apply train_direction
            acc = 1 - acc
        return acc
    
#%% CCS modifications end here

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