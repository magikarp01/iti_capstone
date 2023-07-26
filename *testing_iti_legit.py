#%%
import torch
import numpy as np
import einops
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from tqdm import tqdm
from utils.probing_utils import ModelActs
from utils.model_utils import vicuna_7b, vicuna_13b
from utils.iti_utils import patch_iti
from utils.dataset_utils import MS_Dataset, Elem_Dataset, MisCons_Dataset, Kinder_Dataset, HS_Dataset, BoolQ_Question_Dataset, TruthfulQA_Tfn, CounterFact_Tfn, Fever_Tfn, BoolQ_Tfn, Creak_Tfn, CommonClaim_Tfn, CounterFact_Dataset, TQA_MC_Dataset, EZ_Dataset, TQA_GEN_Dataset
from datasets import Dataset, load_dataset
import random

#%%

model = vicuna_7b(device="cuda")
model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# #%%
# # Define a ModelActs, train probes
# dataset = EZ_Dataset(model.tokenizer)
# acts = ModelActs(model, dataset, act_types=["z"])
# acts.gen_acts(N=1000, store_acts=False)
# acts.train_probes("z", max_iter=1000)

# #%%

# # Actually do ITI
# model.reset_hooks()
# patch_iti(model, acts, topk=10, alpha=0.1, use_probe=True, model_device='cuda')
# model.generate("True or False: The sky is blue.", max_new_tokens = 200)

# #%%

# for dataset in [EZ_Dataset(model.tokenizer), ]:

#     acts = ModelActs(model, dataset, act_types=["z"])
#     acts.gen_acts(N=1000, store_acts=False)
#     acts.train_probes("z", max_iter=1000)

#     model.reset_hooks()
#     patch_iti(model, acts, topk=10, alpha=0.1, use_probe=True, model_device='cuda')
#     print(model.generate("True or False: The sky is blue. Answer:", max_new_tokens = 200))


#%%

# Look at generations when you do ITI, versus ones where you don't for [TruthfulQA, EZ]

# Training dataset -- needs to be dataset_utils..py format
# Testing dataset -- just needs to be a list of strings
def compare_iti_gen(train_dataset, test_dataset):

    model.reset_hooks()

    no_iti_generation = []
    iti_generation = []

    for test in test_dataset:
        gen = model.generate(test, max_new_tokens = 30, do_sample = False, temperature = 1.0, verbose = False).replace("<s>", "").replace("</s>", "").replace(test, "", 1)
        no_iti_generation.append(gen)
        # Add GPT_judge_Score

    acts = ModelActs(model, train_dataset, act_types=["z"])
    acts.gen_acts(N=1000, store_acts=False)
    acts.train_probes("z", max_iter=1000)
    patch_iti(model, acts, topk=10, alpha=0.1, use_probe=True, model_device='cuda')

    for i, test in enumerate(test_dataset):
        gen = model.generate(test, max_new_tokens = 30, do_sample = False, temperature = 1.0, verbose = False).replace("<s>", "").replace("</s>", "").replace(test, "", 1)
        iti_generation.append(gen)
        # Add GPT-Judge score
        print("*** Question --- ", test)
        print("No ITI --- ", no_iti_generation[i])
        print("ITI --- ", iti_generation[i])

#%%

tqa_dataset = TruthfulQA_Tfn(model.tokenizer) # "Is the question below true or false? ... Answer:"
tqa_dataset_2 = TQA_GEN_Dataset(model.tokenizer) # direct truthfulQA generation validation

#%%

tqa_dataset_3 = random.sample(load_dataset("truthful_qa", "generation")['validation']['question'], k=50)



#%%
# Look at performance when you prompt "T" vs "F" for [EZ, Miscons, MS, HS]

compare_iti_gen(tqa_dataset, tqa_dataset_3)


#%%  NOT MY CODE BELOW


random_seed = 5

datanames = ["MS", "Elem", "MisCons", "Kinder", "HS", "TruthfulQA", "CounterFact", "Fever", "Creak", "BoolQ", "CommonClaim"]

ms_data = MS_Dataset(model.tokenizer, questions=True)
elem_data = Elem_Dataset(model.tokenizer, questions=True)
miscons_data = MisCons_Dataset(model.tokenizer, questions=True)
# kinder_data = Kinder_Dataset(model.tokenizer, questions=True)
# hs_data = HS_Dataset(model.tokenizer, questions=True)
# # boolq_data = BoolQ_Question_Dataset(model.tokenizer)

# tqa_data = TruthfulQA_Tfn(model.tokenizer, questions=True)
# cfact_data = CounterFact_Tfn(model.tokenizer, questions=True)
# fever_data = Fever_Tfn(model.tokenizer, questions=True)
# boolq_data = BoolQ_Tfn(model.tokenizer, questions=True)
# creak_data = Creak_Tfn(model.tokenizer, questions=True)
# commonclaim_data = CommonClaim_Tfn(model.tokenizer, questions=True)

# datasets = {"MS": ms_data, "Elem": elem_data, "MisCons": miscons_data, "Kinder": kinder_data, "HS": hs_data, "TruthfulQA": tqa_data, "CounterFact": cfact_data, "Fever": fever_data, "Creak": creak_data, "BoolQ": boolq_data, "CommonClaim": commonclaim_data}
datasets = {"MS": ms_data, "Elem": elem_data, "MisCons": miscons_data}

datanames = datanames[:3]

#%%

n_acts = 1000
acts = {}

for name in datanames:
    # acts[name] = ModelActs(model, datasets[name], act_types=["z", "mlp_out", "resid_post", "resid_pre", "logits"])
    acts[name] = ModelActs(model, datasets[name], act_types=["z", "logits"])
    model_acts: ModelActs = acts[name]
    model_acts.gen_acts(N=n_acts, id=f"{name}_{model_name}_{n_acts}")
    # model_acts.load_acts(id=f"{name}_{model_name}_{n_acts}", load_probes=False)
    model_acts.train_probes("z", max_iter=1000)

#%%

# Plot Truth Score and Info Score when varying topk
from utils.gpt_judge import get_model_generations, get_judge_scores
from utils.iti_utils import patch_heads
from utils.interp_utils import get_head_bools, tot_logit_diff

#%%





