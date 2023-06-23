from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np

# A class for defining our own Dataset classes. Code written by Kevin Wang.

def format_truthfulqa(question, choice):
    return f"Q: {question} A: {choice}"

def tokenized_tqa(dataset, tokenizer): 
    all_prompts = []
    all_labels = []
    for i in range(len(dataset)):
        question = dataset[i]['question']
        choices = dataset[i]['mc2_targets']['choices']
        labels = dataset[i]['mc2_targets']['labels']

        assert len(choices) == len(labels), (len(choices), len(labels))

        for j in range(len(choices)): 
            choice = choices[j]
            label = labels[j]
            prompt = format_truthfulqa(question, choice)
            # if i == 0 and j == 0: 
            #     print(prompt)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(label)
    
    return all_prompts, all_labels

class TQA_MC_Dataset():
    def __init__(self, tokenizer, seed:int = 0):
        self.dataset = load_dataset("truthful_qa", "multiple_choice")['validation']
        self.all_prompts, self.all_labels = tokenized_tqa(self.dataset, tokenizer)
        self.seed = seed
        np.random.seed(seed)
        
    def sample(self, sample_size: int, reset_seed=False):
        if reset_seed: # so that every sample will be the same
            np.random.seed(self.seed)

        # indices = np.random.randint(0, len(self.dataset), size = sample_size)
        indices = np.random.choice(len(self.all_prompts), size = sample_size, replace = False)
        # print(indices) 
        # print(self.all_prompts)
        # print(self.all_labels)
        sample_prompts = []
        sample_labels =[]
        for i in indices:
            sample_prompts.append(self.all_prompts[i])
            sample_labels.append(self.all_labels[i])
        return indices, sample_prompts, sample_labels

        # print(np.array(self.all_prompts)[indices])
        # print(np.array(self.all_labels)[indices])
        # return indices, torch.tensor(self.all_prompts, )[indices], torch.tensor(self.all_labels)[indices]


from datasets import load_dataset
from sklearn.model_selection import train_test_split

def tokenized_tqa_gen(dataset, tokenizer):
    all_prompts = []
    all_labels = []
    all_categories = []
    for i in range(len(dataset)): 
        question = dataset[i]['question']
        category = dataset[i]['category']
        rand_idx = np.random.randint(len(dataset))
        rand_question = dataset[rand_idx]['question']

        for j in range(len(dataset[i]['correct_answers'])): 
            answer = dataset[i]['correct_answers'][j]
            # prompt = format_truthfulqa_end_q(question, answer, rand_question)
            prompt = format_truthfulqa(question, answer)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(1)
            all_categories.append(category)
        
        for j in range(len(dataset[i]['incorrect_answers'])):
            answer = dataset[i]['incorrect_answers'][j]
            # prompt = format_truthfulqa_end_q(question, answer, rand_question)
            prompt = format_truthfulqa(question, answer)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(0)
            all_categories.append(category)
        
    return all_prompts, all_labels, all_categories

class TQA_GEN_Dataset():
    def __init__(self, tokenizer, category: str = "Misconceptions", seed:int = 0):
        full_dataset = load_dataset("truthful_qa", "generation")['validation']
        
        if category is None:
            self.data_set = full_dataset
        else:
            self.dataset = full_dataset.filter(lambda example: example['category'] == category)
    
        self.all_prompts, self.all_labels, self.all_categories = tokenized_tqa_gen(self.dataset, tokenizer)
        
        np.random.seed(seed)
        
    def sample(self, sample_size: int):

        indices = np.random.choice(len(self.all_prompts), size = sample_size, replace = False)

        return indices, np.array(self.all_prompts)[indices], np.array(self.all_labels)[indices]

def format_cfact(question, choice):
    return f"Q: {question} A: {choice}"

def tokenized_cfact(dataset, tokenizer): 
    all_prompts = []
    all_labels = []
    for i in range(len(dataset)):
        prompt = dataset[i]['prompt']
        target_true = dataset[i]['target_true']
        target_false = dataset[i]['target_false']

        true_prompt = prompt + target_true
        true_prompt_toks = tokenizer(true_prompt, return_tensors = 'pt').input_ids
        all_prompts.append(true_prompt_toks)
        all_labels.append(1)
        
        false_prompt = prompt + target_false
        false_prompt_toks = tokenizer(false_prompt, return_tensors = 'pt').input_ids
        all_prompts.append(false_prompt_toks)
        all_labels.append(0)
        
    return all_prompts, all_labels

class CounterFact_Dataset():
    def __init__(self, tokenizer, seed:int = 0):
        self.dataset = load_dataset("NeelNanda/counterfact-tracing")['train']
        self.all_prompts, self.all_labels = tokenized_cfact(self.dataset, tokenizer)
        
        np.random.seed(seed)
        
    def sample(self, sample_size: int):
        indices = np.random.choice(len(self.all_prompts), size = sample_size, replace = False)
        return indices, np.array(self.all_prompts)[indices], np.array(self.all_labels)[indices]


# import pandas as pd

def format_ezdataset(question, choice):
    return f"Q: {question} A: {choice}"

def tokenized_ezdataset(dataset, tokenizer): 
    all_prompts = []
    all_labels = []
    for i in range(len(dataset['train'])):
        prompt = dataset['train'][i]['Question']
        target_true = dataset['train'][i]['Right']
        target_false = dataset['train'][i]['Wrong']

        true_prompt = format_ezdataset(prompt, target_true)
        true_prompt_toks = tokenizer(true_prompt, return_tensors = 'pt').input_ids
        all_prompts.append(true_prompt_toks)
        all_labels.append(1)
        
        false_prompt = format_ezdataset(prompt, target_false)
        false_prompt_toks = tokenizer(false_prompt, return_tensors = 'pt').input_ids
        all_prompts.append(false_prompt_toks)
        all_labels.append(0)
    
    return all_prompts, all_labels

class EZ_Dataset():
    def __init__(self, tokenizer, seed:int = 0):
        self.dataset = load_dataset("csv", data_files = "dumb_facts.csv")
        self.all_prompts, self.all_labels = tokenized_ezdataset(self.dataset, tokenizer)
        np.random.seed(seed)
        
    def sample(self, sample_size: int):
        indices = np.random.choice(len(self.all_prompts), size = sample_size, replace = False)
        sample_prompts = []
        sample_labels =[]
        for i in indices:
            sample_prompts.append(self.all_prompts[i])
            sample_labels.append(self.all_labels[i])
        return indices, sample_prompts, sample_labels

