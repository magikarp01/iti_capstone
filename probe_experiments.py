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

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

import csv



def reformat_acts_for_probing_fully_batched(run_id, N, d_head, n_layers, n_heads, prompt_tag):
    activations_dir = f"{os.getcwd()}/data/large_run_{run_id}/activations"
    load_path = f"{activations_dir}/unformatted"
    save_path = f"{activations_dir}/formatted"

    os.makedirs(save_path, exist_ok=True)

    probe_dataset = torch.zeros((N, n_layers, d_head*n_heads)) #for small dataset and large RAM, just do one load operation
    
    for idx in range(N): 
        if os.path.exists(f"{load_path}/large_run_{run_id}_{prompt_tag}_{idx}.pt"):
            probe_dataset[idx,:,:] = torch.load(f"{load_path}/large_run_{run_id}_{prompt_tag}_{idx}.pt")

    for layer in tqdm(range(n_layers), desc='layer'):
        for head in tqdm(range(n_heads), desc='head', leave=False):
            head_start = head*d_head
            head_end = head_start + d_head
            torch.save(probe_dataset[:,layer,head_start:head_end].squeeze().clone(), f"{save_path}/large_run_{run_id}_{prompt_tag}_l{layer}_h{head}.pt")

def reformat_acts_for_probing_batched_across_heads(run_id, N, d_head, n_layers, n_heads, prompt_tag):
    activations_dir = f"{os.getcwd()}/data/large_run_{run_id}/activations"
    load_path = f"{activations_dir}/unformatted"
    save_path = f"{activations_dir}/formatted"

    os.makedirs(save_path, exist_ok=True)

    probe_dataset = torch.zeros((N, d_head*n_heads)) #if this doesn't fit in memory, don't use batched version
    
    for layer in tqdm(range(n_layers), desc='layer'):
        for idx in range(N): 
            if os.path.exists(f"{load_path}/large_run_{run_id}_{prompt_tag}_{idx}.pt"):
                acts: Float[Tensor, "n_layers d_model"] = torch.load(f"{load_path}/large_run_{run_id}_{prompt_tag}_{idx}.pt")
                probe_dataset[idx,:] = acts[layer,:].squeeze()
        for head in tqdm(range(n_heads), desc='head', leave=False):
            head_start = head*d_head
            head_end = head_start + d_head
            torch.save(probe_dataset[:,head_start:head_end], f"{save_path}/large_run_{run_id}_{prompt_tag}_l{layer}_h{head}.pt")

def reformat_acts_for_probing(run_id, N, d_head, n_layers, n_heads, prompt_tag):
    activations_dir = f"{os.getcwd()}/data/large_run_{run_id}/activations"
    load_path = f"{activations_dir}/unformatted"
    save_path = f"{activations_dir}/formatted"

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
            torch.save(probe_dataset, f"{save_path}/large_run_{run_id}_{prompt_tag}_l{layer}_h{head}.pt")



run_id = 1
N = 2550 #upper bound the global (level 0) index
d_head = 128
n_layers = 80
n_heads = 64
# num_params = "70b"

#reformat_acts_for_probing_fully_batched(run_id, N, d_head, n_layers, n_heads, "honest")
#reformat_acts_for_probing_fully_batched(run_id, N, d_head, n_layers, n_heads, "liar")
#reformat_acts_for_probing_fully_batched(run_id, N, d_head, n_layers, n_heads, "neutral")

# %%

dataset_name = "notrichardren/elem_tf"
dataset = load_dataset(dataset_name)
dataset = dataset["train"].remove_columns(['Unnamed: 0','Topic','Question'])
#MAY NEED TO EDIT TO BE VERY CAREFUL ABOUT INDEXING
loader = DataLoader(dataset, batch_size=1, shuffle=False)
#labels = [batch['Correct'] for batch in loader]
#THIS IS AN EXTREMELY HACKY FIX, REMOVE WHEN USING AN OTHER DATASETS
dont_include = [2477, 2478, 2479, 2480, 2481, 2482, 2483, 2484, 2485]
labels = [batch['Correct'] for batch in loader if not batch['__index_level_0__'] in dont_include]
labels = torch.tensor(labels)




class ModelActsLargeSimple:
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
        y_val_pred_prob = clf.predict(X_test.numpy())
        
        acc = accuracy_score(y_test.numpy(), y_val_pred)

        return clf, acc, y_val_pred_prob

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

    
# %%
acts_honest = ModelActsLargeSimple(1, labels, "honest")
probe_accs_honest, probes_honest, preds_honest = acts_honest.train_z_probes()

acts_liar = ModelActsLargeSimple(1, labels, "liar")
probe_accs_liar, probes_liar, preds_liar = acts_liar.train_z_probes()

acts_neutral = ModelActsLargeSimple(1, labels, "neutral")
probe_accs_neutral, probes_neutral, preds_neutral = acts_neutral.train_z_probes()

fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Create a subplot with 1 row and 3 columns

# Show each tensor in a subplot
axs[0].imshow(probe_accs_honest, vmin=.9, vmax=1, cmap='magma')
axs[1].imshow(probe_accs_neutral, vmin=.9, vmax=1, cmap='magma')
axs[2].imshow(probe_accs_liar, vmin=.9, vmax=1, cmap='magma')

axs[0].set_title("honest")
axs[1].set_title("neutral")
axs[2].set_title("liar")


plt.show()


# %%

# loop through the dataset, get prediction (0 to 1), get probe prob prediction (assume calibrated)
#VALIDATE THAT TORCH RANDOM SPLIT DOES INDICES THE SAME EVERY TIME (only dependence on the dimension)
#edit: fixed by saving indices as an attributes
def get_top_heads(probe_accs, k=10):
    values, indices = torch.topk(probe_accs.view(-1), k)
    indices_2d = (indices // probe_accs.size(1), indices % probe_accs.size(1))
    return list(zip(indices_2d[0].tolist(), indices_2d[1].tolist()))
    #return values, indices_2d


heads_honest = get_top_heads(probe_accs_honest, k=200)
heads_neutral = get_top_heads(probe_accs_neutral, k=200)
heads_liar = get_top_heads(probe_accs_liar, k=200)


labels_honest = labels[acts_honest.test_indices]
labels_neutral = labels[acts_neutral.test_indices]
labels_liar = labels[acts_liar.test_indices]


def get_probe_accs_across_top_heads_by_datapoint(heads_to_include, preds, labels):
    #heads to include List[(Int, Int),...]
    #preds shape (n_layers, n_heads, num_test_points)
    #labels for the test set
    top_heads = torch.zeros((len(heads_to_include), len(labels)))
    for idx, (layer, head) in enumerate(heads_to_include):
        top_heads[idx,:] = preds[layer,head,:]
    accs = (top_heads == labels) #broadcasting
    accs = torch.sum(accs, dim=0)/len(heads_to_include)
    return accs #vector of length test_points
# %%

accs_honest = get_probe_accs_across_top_heads_by_datapoint(heads_honest, preds_honest, labels_honest)
accs_liar = get_probe_accs_across_top_heads_by_datapoint(heads_liar, preds_liar, labels_liar)

# %%

def get_inference_accuracy_calibrated(filename, test_indices, threshold=0):

    calibrated_accs = torch.zeros((N,))
    
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            if idx>0:

                base_idx = int(row[0])

                p_true = float(row[1])
                p_false = float(row[2])
                if p_true > threshold or p_false > threshold:
                    label = int(float(row[3]))
                    
                    pred = p_true/(p_true + p_false)
                    dist = abs(pred - label)
                    acc = 1 - dist
                    calibrated_accs[base_idx] = acc
                    #pred = p_true > p_false
                    #correct = (pred == label) #bool

                    #num_correct += correct
                    #num_total += 1
    mask = (calibrated_accs != 0)

    calibrated_accs = calibrated_accs[mask]
    assert len(calibrated_accs) == len(labels)
    calibrated_accs = calibrated_accs[test_indices]
    return calibrated_accs

# %%
get_inf_file = lambda prompt_tag: f"{os.getcwd()}/data/large_run_1/inference_outputs/inference_output_1_{prompt_tag}_70b.csv"

inf_honest = get_inference_accuracy_calibrated(get_inf_file("honest"), acts_honest.test_indices)
inf_liar = get_inference_accuracy_calibrated(get_inf_file("liar"), acts_liar.test_indices)
#there is no neutral inference accuracy

# %%
fig, ax = plt.subplots()

ax.scatter(accs_honest.tolist(), inf_honest.tolist(), color='blue', alpha=.03, label='honest')
ax.scatter(accs_liar.tolist(), inf_liar.tolist(), color='red', alpha=.03, label='liar')

ax.set_xlabel("probing accuracy")
ax.set_ylabel("inference accuracy")

plt.show()
# %%
plt.hexbin(accs_honest.tolist(), inf_honest.tolist(), gridsize=50, cmap='Blues')
cb = plt.colorbar(label='count in bin')
# %%
plt.hexbin(accs_liar.tolist(), inf_liar.tolist(), gridsize=50, cmap='Reds')
cb = plt.colorbar(label='count in bin')
# %%
plt.hist2d(accs_honest.tolist(), inf_honest.tolist(), bins=(50, 50), cmap=plt.cm.jet)
plt.colorbar()

# %%
