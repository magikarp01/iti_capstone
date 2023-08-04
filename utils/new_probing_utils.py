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
    
    # each of these is a dictionary, keys are act type
    activations = {}
    probes = {}
    probe_accs = {}

    labels: np.array = None
    indices_trains: np.array = None
    indices_tests: np.array = None


    def load_acts(self):
        """
        Load activations into self.activations dictionary. Should be implemented by child classes.
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
        y_val_pred = clf.predict(X_test_head)
        acc = accuracy_score(y_test, y_val_pred)

        return clf, acc


    def train_probes(self, act_type):
        """
        Train probes on all provided activations of act_type in self.activations.
        """
        for probe_index in self.activations[act_type]:
            clf, acc = self._train_probe(act_type, probe_index)
            self.probes[act_type][probe_index] = clf
            self.probe_accs[act_type][probe_index] = acc


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


    def get_transfer_acc(self, act_type, probe_index, data_source: ModelActs):
        """
        Get transfer accuracy of probes trained on these model acts on another set of model acts, on new data. 
        data_source is another ModelActs object that should already have been trained on probes for act_type (with its own train test split).
        """
        
        data_acts = data_source.activations[act_type][probe_index]
        data_labels = data_source.labels[act_type] # test indices from data_source

        probe = self.probes[act_type][probe_index]
        y_pred = probe.predict(data_acts)
        acc = accuracy_score(data_labels, y_pred)

        return acc



class SmallModelActs(ModelActs):
    """
    A class to handle small model activations ran on a user-defined dataset. Class has utilities to generate new activations based on a given model and dataset, and to store those activations for later use using TransformerLens (which is why model has to be small). Not memory efficient, but all-in-one for small models.

    Initialize ModelActs class specifying which activation types to store.
    First, generate acts using gen_acts. Acts can be saved/loaded. Then, train probes on these acts using train_probes: these probes are stored in self.probes dictionary and accuracies are stored in self.probe_accs dictionary.
    """
    def __init__(self, model: HookedTransformer, dataset: Abstract_Dataset, seed = None, act_types = ["z"]):
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


    def gen_acts(self, N = 1000, store_acts = True, filepath = "activations/", id = None, indices=None, storage_device="cpu", verbose=False):
        """
        Automatically generates activations over N samples (returned in self.indices). If store_acts is True, then store in activations folder. Indices are indices of samples in dataset.
        
        
        """        
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

                elif act_type == "attn_scores":
                    # shape of cache.stack_activation(act_type, layer = -1) is (n_l, 1, n_h, s, s)
                    stored_acts = cache.stack_activation(act_type, layer=-1).squeeze().to(device=storage_device)

                else:
                    stored_acts = cache.stack_activation(act_type, layer = -1)[:,0,-1].squeeze().to(device=storage_device)
                    if verbose:
                        print(f"{act_type=}, {cache.stack_activation(act_type, layer = -1).shape=}")
                cached_acts[act_type].append(stored_acts)
        
        # convert lists of tensors into tensors
        stored_acts = {}
        for act_type in self.act_types:
            try:
                stored_acts[act_type] = torch.stack(cached_acts[act_type])
            except: # if cached_acts cannot be stacked, if they are different sizes, probably for attention scores
                stored_acts[act_type] = cached_acts[act_type]

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

    def load_acts(self, id, filepath = "activations/", load_probes=False):
        """
        Loads activations from activations folder. If id is None, then load the most recent activations. 
        If load_probes is True, load from saved probes.picle and all_heads_acc_np.npy files as well.
        """
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






class ModelActsLargeSimple(ModelActs):
    """
    
    """
    """presumes that we already have formatted activations"""
    def __init__(self, run_id, labels, prompt_tag):
        self.run_id = run_id
        self.acts_path = f"{os.getcwd()}/data/large_run_{run_id}/activations/formatted"
        self.labels = labels
        self.prompt_tag = prompt_tag

        self.train_indices = None
        self.test_indices = None

        self.probes = None

    def get_train_test_split(self, X_acts: Float[Tensor, "N d_head"], labels: Float[Tensor, "N"], test_ratio = 0.2):
        probe_dataset = TensorDataset(X_acts, labels)
        if self.train_indices is None and self.test_indices is None:
            generator1 = torch.Generator().manual_seed(42)
            train_data, test_data = random_split(probe_dataset, [1-test_ratio, test_ratio], generator=generator1) 
            self.train_indices = train_data.indices
            self.test_indices = test_data.indices

        X_train, y_train = probe_dataset[self.train_indices]
        X_test, y_test = probe_dataset[self.test_indices]

        return X_train, y_train, X_test, y_test
    

    def _train_single_probe(self, layer, head, max_iter=1000): #Use regularization!!!
        # load_path = f"~/iti_capstone/{filepath}"
        X_acts: Float[Tensor, "N d_head"] = torch.load(f"{self.acts_path}/large_run_{self.run_id}_{self.prompt_tag}_l{layer}_h{head}.pt")
        mask = torch.any(X_acts != 0, dim=1) #mask out zero rows because of difference between global and local indices
        X_acts = X_acts[mask]

        assert X_acts.shape[0] == self.labels.shape[0], "X_acts.shape[0] != self.labels, zero mask fail?"

        X_train, y_train, X_test, y_test = self.get_train_test_split(X_acts, labels) # (train_size, d_head), (test_size, d_head)

        clf = LogisticRegression(max_iter=max_iter).fit(X_train.numpy(), y_train.numpy()) #check shapes
        #also implement plug-and-play pytorch logistic regression and compare performance
        #y_pred = clf.predict(X_train.numpy())

        y_val_pred = clf.predict(X_test.numpy())
        
        acc = accuracy_score(y_test.numpy(), y_val_pred)

        return clf, acc, y_val_pred

    # def get_probe_pred(self, tag):
    # need to implement way to get top preds without storing all preds in large data scenarios
    #     X_acts: Float[Tensor, "N d_head"] = torch.load(f"{self.acts_path}/large_run_{self.run_id}_{self.prompt_tag}_l{layer}_h{head}.pt")
    #     mask = torch.any(X_acts != 0, dim=1) #mask out zero rows because of difference between global and local indices
    #     X_acts = X_acts[mask]

    #     X_train, y_train, X_test, y_test = self.get_train_test_split(X_acts, labels) 

    #     clf = self.probes[tag]

    def train_z_probes(self, max_iter=1000):
        probes = {}
        #preds = np.zeros(n_layers, n_heads, len(self.test_indices)) # this is being incredibly dumb with memory usage, def fix before using larger data
        #accs = {}
        probe_accs = torch.zeros(n_layers, n_heads)
        for layer in tqdm(range(n_layers), desc='layer'): #RELYING ON N_LAYERS AND N_HEADS BEING A GLOBAL HERE
            for head in tqdm(range(n_heads), desc='head', leave=False):
                probe, acc, pred = self._train_single_probe(layer, head, max_iter=max_iter)

                tag = f"l{layer}h{head}"
                probe_accs[layer, head] = acc

                if layer == 0 and head == 0: #hacky
                    preds = torch.zeros((n_layers, n_heads, len(self.test_indices))) # this is being incredibly dumb with memory usage, def fix before using larger data

                preds[layer, head, :] = torch.tensor(pred)

                probes[tag] = probe
        # self.probes = probes
        return probe_accs, probes, preds
        #        probes[tag] = probe
        #        accs[tag] = acc
        #with open(f"large_run_{self.run_id}_probes.pkl", "wb") as file:
        #    pickle.dump(probes, file)
        #with open(f"large_run{self.run_id}_accs.pkl", "wb") as file:
        #    pickle.dump(accs, file)





#todo
#data/probing parallelism
#path handling is atrocious (at the very least abstract it to its own method)

class ModelActsLarge: #separate class for now, can make into subclass later (need to standardize input and output, and manage memory effectively)
    """
    A class that re-implements all the functionality of ModelActs, but optimized for large models. This class doesn't use TransformerLens
    and can thus handle model quantization as well as parallelism. Activation caching is achieved using PyTorch hooks (update: use gen_acts_and_inference_run instead of gen_acts)

    Pipeline:
    1. gen_acts: 
        -(optionally) subsamples from dataset, self.dataset is now TorchSample
        -loops through data, runs model, caches activations and saves in disk labeled 
        as "large_run{id}_{idx}.pt" coupled with "large_run{id}_indices.pt" and/or "large_run{id}_labels.pt" file

    2. reformat_acts_for_probing:
        -reformats "large_run{id}_{idx}.pt" files as "large_run{id}_probe_l{layer}_h{head}.pt"
        -each of these tensors are of shape (N, d_head)

    3. get_train_test_split:
        -given a tensor of shape (N, d_head), returns two tensors of shapes (N_train, )
    """
    def __init__(self, model=None, dataset=None, use_aws=True, seed = None, act_types = ["z"], world_size=1, **kwargs):
        self.model = model #model only needed for gen_acts, otherwise don't load it so as to preserve memory
        self.dataset = dataset

        self.indices = None

        self.act_types = act_types

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.use_aws = use_aws #save data to disk first, then send it all to AWS
        self.world_size = world_size #distributed not supported yet
        self.id = None

        if kwargs["n_layers"] is not None:
            self.n_layers = kwargs["n_layers"] #need to find an adequate way to handle this metadata
            self.d_head = kwargs["d_head"]
            self.n_heads = kwargs["n_heads"]    
            self.d_model = self.d_head * self.n_heads

            self.activation_buffer = torch.zeros((self.n_layers, self.d_model))


    def cache_z_hook_fnc(self, module, input, output, name="", layer_num=0):
        self.activation_buffer[layer_num, :] = input[0][0,-1,:].detach().clone()


    def gen_acts(self, N = 1000, store_acts = True, filepath = "activations/", id = None, indices=None, storage_device="cpu", save_labels=True, distributed=False):
        """
        Doesn't save acts in active memory. Acts may be several hundreds of GB.
        """
        assert store_acts, "If you're using ModelActsLarge, you probably want to store the activations"

        if indices is None:
            indices, prompts, labels = self.dataset.sample(N)
            self.dataset = TorchSample(prompts=prompts, labels=labels, indices=indices)
        self.indices = indices

        self.model.eval()
        hmodel = HookedModule(self.model)
        loader = DataLoader(self.dataset, batch_size=1, shuffle=False) #find dataloader optimizations
        #save_path = f"~/iti_capstone/{filepath}"
        #if not os.path.exists(save_path):
        #    os.system(f"mkdir {save_path}")
        if id is None:
            id = np.random.randint(10000) #BUG: THIS IS DETERMINISTIC
        self.id = id

        #add pytorch hooks
        hook_pairs = []
        if "z" in self.act_types: #only supports hooking for z at the moment
            for layer in range(self.n_layers):
                act_name = f"model.layers.{layer}.self_attn.o_proj"
                hook_pairs.append((act_name, partial(self.cache_z_hook_fnc, name=act_name, layer_num=layer)))
        
        if filepath[-1] != "/":
            filepath = filepath + "/"
        if not os.path.exists(filepath):
            os.system(f"mkdir {filepath}")
        if not os.path.exists(f"{filepath}unformatted"):
            os.system(f"mkdir {filepath}unformatted")
        filepath = filepath + "unformatted/"

        if save_labels:
            torch.save(torch.tensor(self.dataset.all_labels), f"{filepath}large_run{id}_labels.pt")
        if isinstance(self.dataset, TorchSample):
            torch.save(self.indices, f"{filepath}large_run{id}_indices.pt")

        for idx, batch in tqdm(enumerate(loader)):
            token_ids = batch[0].squeeze(dim=0).to(self.model.device)
            with hmodel.hooks(fwd=hook_pairs):
                output = hmodel(token_ids)
            torch.save(self.activation_buffer, f"{filepath}large_run{id}_{idx}.pt")
        #torch.cuda.empty_cache()
        #consider clearing model so can run pipeline all in one swoop

    #note: please make a new folder for this
    def reformat_acts_for_probing(self, id, N=0, filepath="activations/"):
        """
        need to define n_layers, d_head, n_heads, N
        """
        
        if self.dataset is not None:
            N = len(self.dataset)

        if filepath[-1] != "/":
            filepath = filepath + "/"
        load_path = filepath + "unformatted/"
        if not os.path.exists(f"{filepath}formatted"):
            os.system(f"mkdir {filepath}formatted")
        save_path = filepath + "formatted/"

        probe_dataset = torch.empty((N, self.d_head))
        
        for layer in tqdm(range(self.n_layers), desc='layer', ):
            for head in tqdm(range(self.n_heads), desc='head', leave=False):
                head_start = head*self.d_head
                head_end = head_start + self.d_head
                for idx in range(N): #can do smarter things to reduce # of systems calls
                    # O(layers*heads*N)
                    acts: Float[Tensor, "n_layers d_model"] = torch.load(f"{load_path}large_run{id}_{idx}.pt") #saved activation buffers
                    probe_dataset[idx,:] = acts[layer, head_start:head_end].squeeze()
                torch.save(probe_dataset, f"{save_path}large_run{id}_probe_l{layer}_h{head}.pt")

    #def load_acts(self, id, filepath = "activations/", load_probes=False):
    #    raise NotImplementedError
    #def control_for_iti(self, cache_interventions):
    #    raise NotImplementedError
    
    def get_train_test_split(self, X_acts: Float[Tensor, "N d_head"], labels: Float[Tensor, "N"], test_ratio = 0.2):

        probe_dataset = TensorDataset(X_acts, labels)
        generator1 = torch.Generator().manual_seed(42)
        train_data, test_data = random_split(probe_dataset, [1-test_ratio, test_ratio], generator=generator1) #not the same split for every probe

        X_train, y_train = train_data.dataset[train_data.indices]
        X_test, y_test = test_data.dataset[test_data.indicies]

        return X_train, y_train, X_test, y_test
    

    def _train_single_probe(self, id, layer, head, filepath="activations/", max_iter=1000): #Use regularization!!!
        if filepath[-1] != "/":
            filepath = filepath + "/"
        # load_path = f"~/iti_capstone/{filepath}"
        
        X_acts: Float[Tensor, "N d_head"] = torch.load(f"{filepath}large_run{id}_probe_l{layer}_h{head}.pt")

        if self.dataset is not None: #must be same object that called gen_acts!
            labels: Float[Tensor, "N"] = torch.tensor(self.dataset.all_labels)
        else:
            labels: Float[Tensor, "N"] = torch.load(f"{filepath}large_run{id}_labels.pt")

        X_train, y_train, X_test, y_test = self.get_train_test_split(X_acts, labels) # (train_size, d_head), (test_size, d_head)

        clf = LogisticRegression(max_iter=max_iter).fit(X_train.numpy(), y_train.numpy()) #check shapes
        #also implement plug-and-play pytorch logistic regression and compare performance
        y_pred = clf.predict(X_train.numpy())

        y_val_pred = clf.predict(X_test.numpy())

        acc = accuracy_score(y_test.numpy(), y_val_pred)

        return clf, acc


    
    #don't need to store model in memory while doing probing (obviously)
    def _train_probes(self, num_probes, max_iter=1000):
        #can do parallelized probing (see sheet)
        #for now, loop over probes, we can use embarrasing parallelism later
        #write sklearn and pytorch implementation
        #for 
        #probe, acc = self._train_single_probe(id, layer, head, max_iter=max_iter)
        raise NotImplementedError
    
    def train_z_probes(self, id, max_iter=1000):
        probes = {}
        accs = {}
        for layer in tqdm(range(self.n_layers)):
            for head in tqdm(range(self.n_heads)):
                probe, acc = self._train_single_probe(id, layer, head, max_iter=max_iter)
                tag = f"l{layer}h{head}"
                probes[tag] = probe
                accs[tag] = acc
        with open(f"large_run{id}_probes.pkl", "wb") as file:
            pickle.dump(probes, file)
        with open(f"large_run{id}_accs.pkl", "wb") as file:
            pickle.dump(accs, file)
        
    
    def train_mlp_out_probes(self, max_iter=1000):
        #re-create tensors on the fly
        #call train test split
        #call _train_probes
        raise NotImplementedError
    
    def train_probes(self, act_type, max_iter=1000):
        #yes
        raise NotImplementedError
    
    def save_probes(self, id, filepath = "activations/"):
        #super?
        raise NotImplementedError
    
    def get_transfer_acc(self, act_type, data_source: ModelActs):
        #super?
        raise NotImplementedError
    
    def show_top_z_probes(self, topk=50):
        #super?
        raise NotImplementedError