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
from utils.gpt_judge import get_model_generations, get_judge_scores, get_iti_scores

#%%

model = vicuna_7b(device="cuda")
model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#%%

def compare_iti_gen(train_dataset, test_dataset, judge = False):
    """
    Look at generations when you do ITI, versus ones where you don't.

    Training dataset -- needs to be dataset_utils.py format
    Testing dataset -- just needs to be a list of strings
    """

    model.reset_hooks()

    no_iti_generation = []
    iti_generation = []

    for test in test_dataset:
        gen = model.generate(test, max_new_tokens = 30, do_sample = False, temperature = 1.0, verbose = True).replace("<s>", "").replace("</s>", "").replace(test, "", 1).replace("\n", "")
        no_iti_generation.append(gen)
        # Add GPT_judge_Score

    acts = ModelActs(model, train_dataset, act_types=["z"])
    acts.gen_acts(N=1000, store_acts=False)
    acts.train_probes("z", max_iter=1000)
    patch_iti(model, acts, topk=10, alpha=0.1, use_probe=True, model_device='cuda')

    for i, test in enumerate(test_dataset):
        gen = model.generate(test, max_new_tokens = 30, do_sample = False, temperature = 1.0, verbose = False).replace("<s>", "").replace("</s>", "").replace(test, "", 1).replace("\n", "")
        iti_generation.append(gen)
        # Add GPT-Judge score
        print("*** Question --- ", test)
        print("*** No ITI --- ", no_iti_generation[i])
        if judge == True:
            print(f"*** Judge score, true & helpful: {get_judge_scores([no_iti_generation[i]])}")
        print("*** ITI --- ", iti_generation[i])
        if judge == True:
            print(f"*** Judge score, true & helpful: {get_judge_scores([no_iti_generation[i]])}")

#%%

def quantify_iti_mc(train_dataset, test_dataset, labels, k_max = 20):
    """
    Look at MC when you do ITI, versus ones where you don't.

    Training dataset -- needs to be dataset_utils.py format
    Testing dataset -- just needs to be a list of strings
    Labels -- list of booleans
    """

    labels = np.array(labels)
    performance_iti = []
    # performance_no_iti = []

    # no_iti_generation = []
    # no_iti_pred = []
    # for test in test_dataset:
    #     gen = model.generate(test, max_new_tokens = 30, do_sample = False, temperature = 1.0, verbose = True).replace("<s>", "").replace("</s>", "").replace(test, "", 1).replace("\n", "")
    #     no_iti_pred.append(next((w.lower() for w in text.split(r'\s+|[,.:;?!()]') if w.lower() in ["true", "false"]), None) == "true") # check if true comes before false in the string

    #     no_iti_generation.append(gen)
    #     # Add GPT_judge_Score

    # print(f"*** No ITI Accuracy: {np.mean(no_iti_pred == labels)}")

    for k in tqdm(range(k_max)):
        model.reset_hooks()
    
        iti_generation = []
        iti_pred = []
        acts = ModelActs(model, train_dataset, act_types=["z"])
        acts.gen_acts(N=1000, store_acts=False)
        acts.train_probes("z", max_iter=1000)
        patch_iti(model, acts, topk=k, alpha=0.1, use_probe=True, model_device='cuda')

        for i, test in enumerate(test_dataset):
            gen = model.generate(test, max_new_tokens = 5, do_sample = False, temperature = 1.0, verbose = False).replace("<s>", "").replace("</s>", "").replace(test, "", 1).replace("\n", "")
            iti_pred.append(next((w.lower() for w in gen.split(r'\s+|[,.:;?!()]') if w.lower() in ["true", "false"]), None) == "true") # check if true comes before false in the string

            iti_generation.append(gen)
            # Add GPT-Judge score
            print("*** Question --- ", test)
            # print("*** No ITI --- ", no_iti_generation[i])
            print("*** ITI --- ", iti_generation[i])

        iti_pred = np.array(iti_pred)
        # no_iti_pred = np.array(no_iti_pred)

        print(f"*** ITI Accuracy: {np.mean(iti_pred == labels)}")
        performance_iti.append(np.mean(iti_pred == labels))

    print(f"*** ITI Performance: {performance_iti}")
    return performance_iti
        
        
    


        



#%%

tqa_dataset = TruthfulQA_Tfn(model.tokenizer) # "Is the below statement true or false? " + prompt + " Answer:"
# tqa_dataset_2 = TQA_GEN_Dataset(model.tokenizer) # direct truthfulQA generation validation
tqa_dataset_3 = random.sample(load_dataset("truthful_qa", "generation")['validation']['question'], k=50)

# NOTE THAT HERE TEST / TRAIN IS DRAWING FROM SAME DATASET
tqa_dataset_4 = TQA_MC_Dataset(model.tokenizer) # direct truthfulQA MC validation. "Q: ... A: ..."
tqa_dataset_5 = ["Q: " + s + " A:" for s in random.sample(load_dataset("truthful_qa", "multiple_choice")['validation']['question'], k=200)] # direct truthfulQA MC validation. "Q: ... A: ..."


#%%

elem = Elem_Dataset(model.tokenizer) # "Is the below statement true or false? " + prompt + " Answer:"
ms = MS_Dataset(model.tokenizer) # "Is the below statement true or false? " + prompt + " Answer:"
ms_test = ["Is the below statement true or false? " + s + " Answer:" for s in random.sample(load_dataset("notrichardren/ms_tf")["train"]['Question'], k=50)]

#%%
dataset = load_dataset("notrichardren/ms_tf")["train"]

# randomly sample a fraction of the dataset (let's say 10%)
sample_fraction = 0.10
sample_size = int(len(dataset) * sample_fraction)
sample_indices = random.sample(range(len(dataset)), sample_size)
sample = dataset.select(sample_indices)

# extract the 'question' and 'label' columns
question_list = ["Is the below statement true or false? " + sample[i]['Question'] + " Answer:" for i in range(len(sample))]
label_list = [int(sample[i]['Correct']) for i in range(len(sample))] # make it integers

#%%

quantify_iti_mc(elem, question_list, label_list, k_max = 20)

#%%
# Look at performance when you prompt "T" vs "F" for [EZ, Miscons, MS, HS]

# compare_iti_gen(elem, ms_test)


#%%

import matplotlib.pyplot as plt

plt.scatter(range(20), performance_iti)
plt.xlabel("Top K")
plt.ylabel("Accuracy")
plt.title("ITI Performance -- trained on elem dataset")
# %%
