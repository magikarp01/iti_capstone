from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

"""
A class for defining our own Dataset classes for probing and ITI. Code written by Kevin Wang and Phillip Guo.
Datasets have many statements that are either true or false.
""" 

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
    """
    Questions in format "Q: What is underneath the Denver Airport?\nA: The Denver Airport is underneath the city of Denver.
    """
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
        self.dataset = load_dataset("csv", data_files = "datasets/dumb_facts.csv")
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



class Capitals_Dataset():
    """
    Dataset of prompts in the form "Q: What is the capital of {str(country)}? A: {capital}"
    capital may be correct or incorrect capital of country.
    Notable: dataset has 248*2=496 examples, much fewer than others.
    """
    def load_dataset(self, tokenizer, csv_file, wrong_seed=5, omit_capital=False):
        """
        Load Pandas dataframe from world_capitals.csv, then generate dataset of prompts (either true or
        false) in form "Q: What is the capital of {str(country)}? A: {capital}".
        If omit_capital is provided, then don't add (correct or incorrect) capital to prompt. Returned labels are meaningless if this option is True.
        Wrong seed is used to generate incorrect capital for false prompts.
        """
        dataframe = pd.read_csv(csv_file)

        prompts = []
        labels = []

        np.random.seed(wrong_seed) # so that wrong capital is consistent or replicable
        for idx in range(dataframe.shape[0]):

            if omit_capital:
                country = dataframe.at[idx, 'country']
                prompts.append(f"Q: What is the capital of {str(country)}? A:")
            else:
                country = dataframe.at[idx, 'country']
                capital = dataframe.at[idx, 'capital']
                prompts.append(f"Q: What is the capital of {str(country)}? A: {capital}")
                labels.append(1)

                wrong_capital = dataframe.at[np.random.randint(dataframe.shape[0]), 'capital']
                while wrong_capital == capital: # regenerate until not equal to correct capital
                    wrong_capital = dataframe.at[np.random.randint(dataframe.shape[0]), 'capital']

                prompts.append(f"Q: What is the capital of {str(country)}? A: {wrong_capital}")
                labels.append(0)   

        tokenized_prompts = []
        for prompt in prompts:
            tokenized_prompts.append(tokenizer(prompt, return_tensors = 'pt').input_ids)
        return tokenized_prompts, labels

    def __init__(self, tokenizer, seed:int = 0):
        # self.dataset = load_dataset("csv", data_files = "world_capitals.csv")
        self.all_prompts, self.all_labels = self.load_dataset(tokenizer, "datasets/world_capitals.csv", wrong_seed=5)

        np.random.seed(seed)
        
    def sample(self, sample_size: int):
        indices = np.random.choice(len(self.all_prompts), size = sample_size, replace = False)
        sample_prompts = []
        sample_labels =[]
        for i in indices:
            sample_prompts.append(self.all_prompts[i])
            sample_labels.append(self.all_labels[i])
        return indices, sample_prompts, sample_labels
    
    def sample_questions(self, sample_size: int, seed=None):
        """
        Alternative sample function to sample questions without the capital given.
        This allows us to test causal interventions, seeing if ITI actually does improve response rate.
        No labels are returned.
        """
        question_prompts, _ = self.load_dataset("world_capitals.csv", omit_capital=True)

        if seed is not None:
            np.random.seed(seed)
        
        indices = np.random.choice(len(question_prompts), size = sample_size, replace = False)
        sample_prompts = []
        for i in indices:
            sample_prompts.append(question_prompts[i])
        return indices, sample_prompts

class BoolQ_Dataset():
    def __init__(self, tokenizer, seed:int = 0):
        self.dataset = load_dataset("boolq")
        # self.all_prompts, self.all_labels = tokenized_boolq(self.dataset, tokenizer)
        
        np.random.seed(seed)