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


def reformat_acts_for_probing(run_id, N, d_head, n_layers, n_heads, num_params, prompt_tag):
    activations_dir = f"{os.getcwd()}/data/large_run_{run_id}/activations"
    load_path = f"{activations_dir}/unformatted"
    save_path = f"{activations_dir}/formatted"

    if not os.path.exists(save_path):
        os.system(f"mkdir {save_path}")

    probe_dataset = torch.empty((N, d_head))
    
    for layer in tqdm(range(n_layers), desc='layer', ):
        for head in tqdm(range(n_heads), desc='head', leave=False):
            head_start = head*d_head
            head_end = head_start + d_head
            for idx in range(N): #can do smarter things to reduce # of systems calls; O(layers*heads*N)
                if os.path.exists(f"{load_path}/large_run_{run_id}_{num_params}_{prompt_tag}_{idx}.pt"): #quick patch
                    acts: Float[Tensor, "n_layers d_model"] = torch.load(f"{load_path}/large_run_{run_id}_{num_params}_{prompt_tag}_{idx}.pt") #saved activation buffers
                    probe_dataset[idx,:] = acts[layer, head_start:head_end].squeeze()
            torch.save(probe_dataset, f"{save_path}/large_run_{run_id}_{num_params}_{prompt_tag}_l{layer}_h{head}.pt")



#RUN ID HAS BEEN TAKEN OUT OF LOAD
run_id = 1
N = 2500 #upper bound the global (level 0) index
d_head = 128
n_layers = 80
n_heads = 64
num_params = "70b"

#reformat_acts_for_probing(run_id, N, d_head, n_layers, n_heads, num_params, "honest")
#reformat_acts_for_probing(run_id, N, d_head, n_layers, n_heads, num_params, "liar")
#reformat_acts_for_probing(run_id, N, d_head, n_layers, n_heads, num_params, "neutral")


dataset_name = "notrichardren/elem_tf"
dataset = load_dataset(dataset_name)
dataset = dataset["train"].remove_columns(['Unnamed: 0','Topic','Question'])
#MAY NEED TO EDIT TO BE VERY CAREFUL ABOUT INDEXING
loader = DataLoader(dataset, batch_size=1, shuffle=False)
labels = [batch['Correct'] for batch in loader]
labels = torch.tensor(labels)




class ModelActsLargeSimple:
    """presumes that we already have formatted activations"""
    def __init__(self, run_id, labels, prompt_tag):
        self.run_id = run_id
        self.acts_path = f"{os.getcwd()}/data/large_run_{run_id}/activations/formatted"
        self.labels = labels
        self.prompt_tag = prompt_tag

    def get_train_test_split(self, X_acts: Float[Tensor, "N d_head"], labels: Float[Tensor, "N"], test_ratio = 0.2):
        probe_dataset = TensorDataset(X_acts, labels)
        generator1 = torch.Generator().manual_seed(42)
        train_data, test_data = random_split(probe_dataset, [1-test_ratio, test_ratio], generator=generator1) #not the same split for every probe

        X_train, y_train = train_data.dataset[train_data.indices]
        X_test, y_test = test_data.dataset[test_data.indices]

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

        return clf, acc


    #don't need to store model in memory while doing probing (obviously)
    def _train_probes(self, num_probes, max_iter=1000):
        #can do parallelized probing (see sheet)
        #for now, loop over probes, we can use embarrasing parallelism later
        #write sklearn and pytorch implementation
        #for 
        #probe, acc = self._train_single_probe(id, layer, head, max_iter=max_iter)
        raise NotImplementedError
    
    def train_z_probes(self, max_iter=1000):
        #probes = {}
        #accs = {}
        probe_accs = torch.empty(n_layers, n_heads)
        for layer in tqdm(range(n_layers)): #RELYING ON N_LAYERS AND N_HEADS BEING A GLOBAL HERE
            for head in tqdm(range(n_heads)):
                probe, acc = self._train_single_probe(layer, head, max_iter=max_iter)
                tag = f"l{layer}h{head}"
                probe_accs[layer, head] = acc
        return probe_accs
        #        probes[tag] = probe
        #        accs[tag] = acc
        #with open(f"large_run_{self.run_id}_probes.pkl", "wb") as file:
        #    pickle.dump(probes, file)
        #with open(f"large_run{self.run_id}_accs.pkl", "wb") as file:
        #    pickle.dump(accs, file)

    
