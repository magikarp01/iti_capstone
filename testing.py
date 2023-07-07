# %%
import os
from utils.model_loading_utils import load_llama
from utils.probing_utils import ModelActsLarge
from utils.dataset_utils import TorchDataset
import time


class TCM:
    """context manager for timing code segments"""
    def __init__(self):
        self.start_time = 0

    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Time: ", time.time() - self.start_time)


with TCM():
    model = load_llama("stable-vicuna-13b", os.getcwd())
    dataset = TorchDataset(model.tokenizer,"tqa")
    kwargs = {"n_layers" : 40,
                "d_head" : 128,
                "n_heads" : 40,
                "d_model" : 5120
                }
    acts = ModelActsLarge(model, dataset, **kwargs)
    acts.gen_acts(N=50)
    acts.reformat_acts_for_probing(5986, N=50)
    acts.train_z_probes(acts.id)