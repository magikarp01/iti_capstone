from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm import tqdm



def get_balanced_indices(indices, labels, sample_size):
    """
    A sampling method for getting a balanced sample of indices from a list of indices and labels, where the labels are unbalanced (either 0 or 1).
    """
    pairs = list(zip(indices, labels))

    # Split the pairs into two lists depending on the label
    pairs_0 = [pair[0] for pair in pairs if pair[1] == 0]
    pairs_1 = [pair[0] for pair in pairs if pair[1] == 1]

    # Get the size of the smaller list
    min_size = min(len(pairs_0), len(pairs_1))

    assert sample_size//2 <= min_size

    # Randomly select 'min_size' elements from each list
    sampled_indices_0 = np.random.choice(len(pairs_0), sample_size//2, replace=False)
    sampled_indices_1 = np.random.choice(len(pairs_1), sample_size//2, replace=False)

    sampled_pairs_0 = [pairs_0[i] for i in sampled_indices_0]
    sampled_pairs_1 = [pairs_1[i] for i in sampled_indices_1]

    # Merge the lists and extract the indices
    balanced_indices = np.concatenate((sampled_pairs_0, sampled_pairs_1))

    np.random.shuffle(balanced_indices)
    return balanced_indices


class Abstract_Dataset:
    """
    A class for defining our own Dataset classes for probing and ITI. Code written by Kevin Wang and Phillip Guo.
    Datasets have many statements that are either true or false. They need a sample method that returns random
    indices of statements in the dataset, the tokenized statements, and labels.
    Their init method needs to define self.all_prompts (integer tokens) and self.all_labels.
    """ 
    def __init__(self, tokenizer, seed:int = 0):
        raise NotImplementedError

    def sample(self, sample_size: int, reset_seed=False, balanced=True):
        """
        indices is of type numpy array
        sample_prompts is of type List of Tensors
        sample_labels is of type List of Ints
        """
        if reset_seed:
            np.random.seed(self.seed)
        
        if balanced:
            indices = get_balanced_indices(range(len(self.all_prompts)), self.all_labels, sample_size)
        else: 
            indices = np.random.choice(len(self.all_prompts), size = sample_size, replace = False)
        
        sample_prompts = []
        sample_labels =[]
        for i in indices:
            sample_prompts.append(self.all_prompts[i])
            sample_labels.append(self.all_labels[i])
        return indices, sample_prompts, sample_labels



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

class TQA_MC_Dataset(Abstract_Dataset):
    """
    Questions in format "Q: What is underneath the Denver Airport?\nA: The Denver Airport is underneath the city of Denver.
    """
    def __init__(self, tokenizer, seed:int = 0):
        self.dataset = load_dataset("truthful_qa", "multiple_choice")['validation']
        self.all_prompts, self.all_labels = tokenized_tqa(self.dataset, tokenizer)
        self.seed = seed
        np.random.seed(seed)
    

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

class TQA_GEN_Dataset(Abstract_Dataset):
    def __init__(self, tokenizer, category: str = "Misconceptions", seed:int = 0):
        full_dataset = load_dataset("truthful_qa", "generation")['validation']
        
        if category is None:
            self.data_set = full_dataset
        else:
            self.dataset = full_dataset.filter(lambda example: example['category'] == category)
    
        self.all_prompts, self.all_labels, self.all_categories = tokenized_tqa_gen(self.dataset, tokenizer)
        
        np.random.seed(seed)
        

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

class CounterFact_Dataset(Abstract_Dataset):
    def __init__(self, tokenizer, seed:int = 0):
        self.dataset = load_dataset("NeelNanda/counterfact-tracing")['train']
        self.all_prompts, self.all_labels = tokenized_cfact(self.dataset, tokenizer)
        
        np.random.seed(seed)
        

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

class EZ_Dataset(Abstract_Dataset):
    def __init__(self, tokenizer, seed:int = 0):
        self.dataset = load_dataset("csv", data_files = "datasets/dumb_facts.csv")
        self.all_prompts, self.all_labels = tokenized_ezdataset(self.dataset, tokenizer)
        np.random.seed(seed)


class Capitals_Dataset(Abstract_Dataset):
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



def TF_helper(prompts, labels, tokenizer):
    """
    Helper function for ChatGPTGen_Dataset.
    """
    all_prompts = []
    all_labels = []
    for i in range(len(prompts)):

        prompt = prompts[i]
        label = labels[i]

        prompt_true = "True or False: " + prompt + " True"
        all_prompts.append(tokenizer(prompt_true, return_tensors = 'pt').input_ids)
        if label == True:
            all_labels.append(1)
        else:
            all_labels.append(0)
        
        prompt_false = "True or False: " + prompt + " False"
        all_prompts.append(tokenizer(prompt_false, return_tensors = 'pt').input_ids)
        if label == False:
            all_labels.append(0)
        else:
            all_labels.append(1)
    
    return all_prompts, all_labels


def TF_questions_helper(prompts, labels, tokenizer, custom_prompt=None):
    """
    Helper function for ChatGPTGen_Dataset.
    Does not include "True" or "False" in the prompt, label is instead whether or not statement is actually true.
    custom_prompt does not work yet.
    """
    all_prompts = []
    all_labels = []
    for i in range(len(prompts)):

        prompt = prompts[i]
        label = labels[i]

        prompt_true = "Is the below statement true or false? " + prompt + " Answer:"
        all_prompts.append(tokenizer(prompt_true, return_tensors = 'pt').input_ids)
        all_labels.append(label)
    
    return all_prompts, all_labels


class ChatGPTGen_Dataset(Abstract_Dataset):
    def __init__(self, tokenizer, seed:int = 0, questions=False, custom_prompt=None):
        """
        If questions = True, then load dataset of prompts in the form "True or False: {str(prompt)}" without True or False at end. Else, load in normal "True or False: {str(prompt)} True/False" format.
        """
        # define self.dataset in child class
        assert self.dataset is not None
        if questions:
            self.all_prompts, self.all_labels = TF_questions_helper(self.dataset['Question'], self.dataset['Correct'], tokenizer, custom_prompt)
        else:
            self.all_prompts, self.all_labels = TF_helper(self.dataset['Question'], self.dataset['Correct'], tokenizer)
        self.tokenizer = tokenizer
        self.seed = np.random.seed(seed)


class MS_Dataset(ChatGPTGen_Dataset):
    def __init__(self, *args, **kwargs):
        self.dataset = load_dataset("notrichardren/elem_tf")["train"]
        super().__init__(*args, **kwargs)

class Elem_Dataset(ChatGPTGen_Dataset):
    def __init__(self, *args, **kwargs):
        self.dataset = load_dataset("notrichardren/ms_tf")["train"]
        super().__init__(*args, **kwargs)

class MisCons_Dataset(ChatGPTGen_Dataset):
    def __init__(self, *args, **kwargs):
        self.dataset = load_dataset("notrichardren/misconceptions_tf")["train"]
        super().__init__(*args, **kwargs)

class Kinder_Dataset(ChatGPTGen_Dataset):
    def __init__(self, *args, **kwargs):
        self.dataset = load_dataset("notrichardren/kindergarten_tf")["train"]
        super().__init__(*args, **kwargs)

class HS_Dataset(ChatGPTGen_Dataset):
    def __init__(self, *args, **kwargs):
        self.dataset = load_dataset("notrichardren/hs_tf")["train"]
        super().__init__(*args, **kwargs)


class ChatGPTGen_Dataset_Truthfulness(Abstract_Dataset):
    def __init__(self, tokenizer, seed:int = 0, questions=False, custom_prompt=None):
        """
        If questions = True, then load dataset of prompts in the form "True or False: {str(prompt)}" without True or False at end. Else, load in normal "True or False: {str(prompt)} True/False" format.
        """
        # define self.dataset in child class
        assert self.dataset is not None
        if questions:
            self.all_prompts, self.all_labels = TF_questions_helper(self.dataset['claim'], self.dataset['label'], tokenizer, custom_prompt)
        else:
            self.all_prompts, self.all_labels = TF_helper(self.dataset['claim'], self.dataset['label'], tokenizer)
        self.tokenizer = tokenizer
        self.seed = np.random.seed(seed)

class TruthfulQA_Tfn(ChatGPTGen_Dataset_Truthfulness):
    def __init__(self, *args, **kwargs):
        string_list = ["TruthfulQA"]
        dataset = load_dataset("notrichardren/truthfulness")["train"]
        df = dataset.to_pandas()
        df = df[df['origin_dataset'].isin(string_list)]
        self.dataset = Dataset.from_pandas(df)
        super().__init__(*args, **kwargs)

class CounterFact_Tfn(ChatGPTGen_Dataset_Truthfulness):
    def __init__(self, *args, **kwargs):
        string_list = ["CounterFact"]
        dataset = load_dataset("notrichardren/truthfulness")["train"]
        df = dataset.to_pandas()
        df = df[df['origin_dataset'].isin(string_list)]
        self.dataset = Dataset.from_pandas(df)
        super().__init__(*args, **kwargs)

class Fever_Tfn(ChatGPTGen_Dataset_Truthfulness):
    def __init__(self, *args, **kwargs):
        string_list = ["fever_v1.0_labelleddev", "fever_v1.0_train","fever_v2.0"]
        dataset = load_dataset("notrichardren/truthfulness")["train"]
        df = dataset.to_pandas()
        df = df[df['origin_dataset'].isin(string_list)]
        self.dataset = Dataset.from_pandas(df)
        super().__init__(*args, **kwargs)

class BoolQ_Tfn(ChatGPTGen_Dataset_Truthfulness):
    def __init__(self, *args, **kwargs):
        string_list = ["boolq_train", "boolq_test"]
        dataset = load_dataset("notrichardren/truthfulness")["train"]
        df = dataset.to_pandas()
        df = df[df['origin_dataset'].isin(string_list)]
        self.dataset = Dataset.from_pandas(df)
        super().__init__(*args, **kwargs)

class Creak_Tfn(ChatGPTGen_Dataset_Truthfulness):
    def __init__(self, *args, **kwargs):
        string_list = ["creak_train", "creak_dev", "creak_contrast_set"]
        dataset = load_dataset("notrichardren/truthfulness")["train"]
        df = dataset.to_pandas()
        df = df[df['origin_dataset'].isin(string_list)]
        self.dataset = Dataset.from_pandas(df)
        super().__init__(*args, **kwargs)

class CommonClaim_Tfn(ChatGPTGen_Dataset_Truthfulness):
    def __init__(self, *args, **kwargs):
        string_list = ["CommonClaim"]
        dataset = load_dataset("notrichardren/truthfulness")["train"]
        df = dataset.to_pandas()
        df = df[df['origin_dataset'].isin(string_list)]
        self.dataset = Dataset.from_pandas(df)
        super().__init__(*args, **kwargs)


class BoolQ_Dataset(Abstract_Dataset):
    """
    Dataset of True/False statements. Statements consist of "true or false: {question}. A: {true or false}". 
    For some reason, dataset is all in lowercase, may degrade performance.
    18854 examples if train=True, 6540 if train=False.
    """
    def __init__(self, tokenizer, seed:int = 0, train=True):
        self.dataset = load_dataset("boolq")["train" if train else "validation"]
        # self.all_prompts, self.all_labels = tokenized_boolq(self.dataset, tokenizer)
        
        prompts = []
        labels = []
        for idx, question in enumerate(self.dataset['question']):
            prompt = f"true or false: {question}? A:"
            prompts.append(tokenizer(prompt + " true", return_tensors='pt').input_ids)
            prompts.append(tokenizer(prompt + " false", return_tensors='pt').input_ids)

            if self.dataset['answer'][idx]:
                labels.append(1)
                labels.append(0)
            else:
                labels.append(0)
                labels.append(1)
        np.random.seed(seed)

        self.all_prompts = prompts
        self.all_labels = labels

class BoolQ_Question_Dataset(Abstract_Dataset):
    """
    Dataset of questions, without the " true" or " false" at end of statement given. Tokenized.
    """
    def __init__(self, tokenizer, seed:int = 0, train=True):
        self.dataset = load_dataset("boolq")["train" if train else "validation"]
        
        prompts = []
        labels = []
        prompt_start = "Is this true or false:"
        for idx, question in enumerate(self.dataset['question']):
            prompt = f"{prompt_start} {question}? A:"
            prompts.append(tokenizer(prompt, return_tensors='pt').input_ids)

            if self.dataset['answer'][idx] == True:
                labels.append(1)
            else:
                labels.append(0)
        np.random.seed(seed)

        self.all_prompts = prompts
        self.all_labels = labels

#%%

class CCS_Dataset:
    """
    Trying to create a dataset that creates things in *pairs*
    """
    def __init__(self, label_dict, format_prompt, dataset, tokenizer, dataset_name = "imdb", seed:int = 0):
        self.tokenizer = tokenizer
        self.dataset = dataset # HuggingFace dataset
        self.label_dict = label_dict
        self.format_prompt = format_prompt
        self.dataset_name = dataset_name

    def sample_pair(self, sample_size: int, reset_seed=False, balanced=True, used_idx=[], max_token_length = None): # WILL REFACTOR THIS & CLEAN IT UP LATER
        """
        indices is of type numpy array
        sample_prompts is of type List of Tensors
        sample_labels is of type List of Ints

        Balanced by default; unbalanced implementation hasn't happened yet lol
        """

        all_neg_hs, all_pos_hs, all_gt_labels = [], [], []

        # Length requirement
        max_token_length = self.tokenizer.model_max_length if max_token_length is None else max_token_length

        # BALANCE
        neg_labels = 0
        pos_labels = 0
        neg_labels_limit = sample_size//2
        pos_labels_limit = sample_size-neg_labels_limit

        # keep track
        # print(f"intial used_idx: {used_idx}")
        used_idxs = used_idx
        visited_idxs = used_idx
        toolong_idxs = []

        for _ in tqdm(range(sample_size)):
            while True:
                idx = np.random.randint(len(self.dataset))
                if idx in visited_idxs:
                    continue
                visited_idxs.append(idx)

                # Find all the variables relevant to this prompt
                text, true_label = self.dataset[idx]["text"], self.dataset[idx]["label"]
                try:
                    text1 = self.dataset[idx]["text1"]
                except:
                    text1 = ""
                try:
                    text2 = self.dataset[idx]["text2"]
                except:
                    text2 = ""
                pos_prompt = self.format_prompt(true_label, text, text1, text2, self.dataset_name)

                # Length requirement (not met)
                if not (len(self.tokenizer(pos_prompt)['input_ids']) < (max_token_length - 20)):
                    toolong_idxs.append(idx)
                    continue
                # Balance requirement (met) + length requirement met
                elif ((pos_labels < pos_labels_limit and true_label == 1) or (neg_labels < neg_labels_limit and true_label == 0)):
                    if true_label == 0:
                        neg_labels += 1
                    else:
                        pos_labels += 1
                    break
            for i, label in enumerate(self.label_dict.get(self.dataset_name)):
                if i != true_label:
                    neg_prompt = self.format_prompt(i, text, text1, text2, self.dataset_name)
                    all_neg_hs.append(neg_prompt)
                    all_pos_hs.append(pos_prompt)
                    all_gt_labels.append(true_label)


        # neg_p = np.stack(all_neg_hs)
        # pos_p = np.stack(all_pos_hs)
        # y = np.stack(all_gt_labels)

        return all_neg_hs, all_pos_hs, all_gt_labels, used_idxs # prompt_no, prompt_yes, y, used_idxs # TODO rename