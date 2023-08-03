# %%
import os
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle

from datasets import load_dataset


def reformat_acts_for_probing(run_id, N, d_head, n_layers, n_heads, prompt_tag):
    activations_dir = f"{os.getcwd()}/data/large_run_{run_id}/activations"
    load_path = f"{activations_dir}/unformatted"
    save_path = f"{activations_dir}/formatted_test"

    if not os.path.exists(save_path):
        os.system(f"mkdir {save_path}")

    probe_dataset = torch.zeros((N, d_head))
    
    for layer in tqdm(range(n_layers), desc='layer', ):
        for head in tqdm(range(n_heads), desc='head', leave=False):
            head_start = head*d_head
            head_end = head_start + d_head
            for idx in range(N): #can do smarter things to reduce # of systems calls; O(layers*heads*N)
                if os.path.exists(f"{load_path}/large_run_{run_id}_{prompt_tag}_{idx}.pt"): #quick patch
                    acts: Float[Tensor, "n_layers d_model"] = torch.load(f"{load_path}/large_run_{run_id}_{prompt_tag}_{idx}.pt") #saved activation buffers
                    probe_dataset[idx,:] = acts[layer, head_start:head_end].squeeze()
            print(probe_dataset)
            torch.save(probe_dataset, f"{save_path}/large_run_{run_id}_{prompt_tag}_l{layer}_h{head}.pt")



run_id = 1
N = 2550 #upper bound the global (level 0) index
d_head = 128
#n_layers = 80
#n_heads = 64
n_layers = 1
n_heads = 1


reformat_acts_for_probing(run_id, N, d_head, n_layers, n_heads, "honest")
reformat_acts_for_probing(run_id, N, d_head, n_layers, n_heads, "liar")
reformat_acts_for_probing(run_id, N, d_head, n_layers, n_heads, "neutral")











# %%

def apply_mask(probe_dataset):
    mask = torch.any(probe_dataset != 0, dim=1)
    X_acts = probe_dataset[mask]
    return X_acts












"""
import os
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle

from datasets import load_dataset

run_id = 1
N = 2550
d_head = 128
n_layers = 80
n_heads = 64
num_params = "70b"

prompt_tag = "neutral"

head = 0
layer = 0

activations_dir = f"{os.getcwd()}/data/large_run_{run_id}/activations"
load_path = f"{activations_dir}/unformatted"
save_path = f"{activations_dir}/formatted"

if not os.path.exists(save_path):
    os.system(f"mkdir {save_path}")


probe_dataset = torch.empty((N, d_head))

head_start = head*d_head
head_end = head_start + d_head
for idx in range(N): #can do smarter things to reduce # of systems calls; O(layers*heads*N)
    if os.path.exists(f"{load_path}/large_run_{run_id}_{prompt_tag}_{idx}.pt"): #quick patch
        acts: Float[Tensor, "n_layers d_model"] = torch.load(f"{load_path}/large_run_{run_id}_{prompt_tag}_{idx}.pt") #saved activation buffers
        probe_dataset[idx,:] = acts[layer, head_start:head_end].squeeze()


mask = torch.any(probe_dataset != 0, dim=1) #mask out zero rows because of difference between global and local indices
X_acts = probe_dataset[mask]
"""
