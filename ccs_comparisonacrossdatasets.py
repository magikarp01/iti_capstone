#%%
from tqdm import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from promptsource.templates import DatasetTemplates
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression

device = "cuda:0"
np.random.seed(42) # all randomness comes from numpy
models_downloaded = False # set to True if you have already downloaded the models and they're in this repo's directory.speeds up downloading

#%% Load datasets

dataset_names = [("imdb",), ("amazon_polarity",), ("ag_news",), ("dbpedia_14",), ("super_glue", "copa"),
                 ("super_glue", "rte"), ("boolq",), ("glue", "qnli"), ("piqa",), ("chenxwh/gen-storycloze",)]
dataset_names_singular = ["imdb", "amazon_polarity", "ag_news", "dbpedia_14", "copa", "rte", "boolq", "qnli", "piqa", "story-cloze"]
label_dict = {
    "imdb": ["negative", "positive"], # This is for normal IMDB
    "amazon_polarity": ["negative", "positive"],
    "ag_news": ["politics", "sports", "business", "technology"],
    "dbpedia_14": ["company", "educational institution", "artist", "athlete", "office holder", "mean of transportation", "building", "natural place", "village", "animal",  "plant",  "album",  "film",  "written work"],
    "copa": ["choice 1", "choice 2"],
    "rte": ["yes", "no"],   # whether entail
    "boolq": ["false", "true"],
    "qnli": ["yes", "no"],  # represent whether entail
    "piqa": ["solution 1", "solution 2"],
    "story-cloze": ["choice 1", "choice 2"],
}
datasets = {}

for i, dataset_tuple in enumerate(dataset_names):
    dataset = load_dataset(*dataset_tuple)["train"]

    if dataset_names_singular[i] == "amazon_polarity" or dataset_names_singular[i] == "dbpedia_14":
        df = dataset.to_pandas()
        df["text"] = "[" + df["title"] + "] " + df["content"]
        dataset = Dataset.from_pandas(df)
    elif dataset_names_singular[i] == "copa":
        dataset2 = load_dataset(*dataset_tuple)["validation"]
        df_concat = pd.concat([pd.DataFrame(dataset), pd.DataFrame(dataset2)])
        dataset = Dataset.from_pandas(df_concat)
        dataset = dataset.rename_column('choice1', 'text1')
        dataset = dataset.rename_column('choice2', 'text2')
        dataset = dataset.rename_column('premise', 'text')
    elif dataset_names_singular[i] == "rte":
        dataset = dataset.rename_column('premise', 'text')
        dataset = dataset.rename_column('hypothesis', 'text1')
    elif dataset_names_singular[i] == "boolq":
        dataset = dataset.rename_column('question', 'text1')
        dataset = dataset.rename_column('passage', 'text')
        dataset = dataset.rename_column('answer', 'label')
        dataset = dataset.to_pandas()
        dataset['label'] = dataset['label'].astype(int)
        dataset = Dataset.from_pandas(dataset)
    elif dataset_names_singular[i] == "qnli":
        dataset = dataset.rename_column('question', 'text')
        dataset = dataset.rename_column('sentence', 'text1')
    elif dataset_names_singular[i] == "piqa":
        dataset = dataset.rename_column('goal', 'text')
        dataset = dataset.rename_column('sol1', 'text1')
        dataset = dataset.rename_column('sol2', 'text2')
    elif dataset_names_singular[i] == "story-cloze":
        dataset = dataset.rename_column('context', 'text')
        dataset = dataset.rename_column('sentence_quiz1', 'text1')
        dataset = dataset.rename_column('sentence_quiz2', 'text2')

    # add dataset_names_Singular as key and dataset as value to datasets
    datasets.update({dataset_names_singular[i]: dataset})

    dataset_name = "/".join(dataset_tuple)
    print("Loaded dataset:", dataset_name)
    print(dataset)

    # all_prompts = DatasetTemplates(dataset_name)
    # print(all_prompts)



#%%
def format_prompt(label, text, text1, text2, dataset_name = "imdb"):
    """
    Given an imdb example ("text") and corresponding label (0 for negative, or 1 for positive), 
    returns a zero-shot prompt for that example (which includes that label as the answer).
    
    (This is just one example of a simple, manually created prompt.)
    """
    if dataset_name == "imdb":
        return "The following movie review expresses a " + label_dict[dataset_name][label] + " sentiment:\n" + text
    if dataset_name == "amazon_polarity":
        return "The following Amazon review expresses a " + label_dict[dataset_name][label] + " sentiment:\n" + text
        # text = title and content
    if dataset_name == "ag_news":
        return "The topic of the following news article is about " + label_dict[dataset_name][label] + ":\n" + text
    if dataset_name == "dbpedia_14":
        return "The following article relates to " + label_dict[dataset_name][label] + "s:\n" + text
        # text = title and content
    if dataset_name == "copa":
        return f'Story: {text} \nIn this story, out of "{text1}" and "{text2}", the sentence is most likely to follow is {["the former", "the latter"][label]}.'
        # text = premise. text1 and text2 are choice1 choice2
    if dataset_name == "rte":
        return f"Passage: {text}\nQuestion: Does this imply that {text1}?\nAnswer here: {['Yes', 'No'][label]}."
        # text = premise
        # text1 = hypothesis
    if dataset_name == "boolq":
        return f"{text}\nQuestion: {text1}? {['Yes', 'No'][label]}"
        # text = passage
        # text1 = question
    if dataset_name == "qnli":
        return f"Question: {text}\nAnswer: {text1}\nDoes the information in the provided answer help completely the question? {['Yes', 'No'][label]}"
        # text = question
        # text1 = answer
    if dataset_name == "piqa":
        return f"Which choice makes the most sense? \nQuestion: {text}\nChoice 1: {text1}\nChoice 2: {text2}\nAnswer here: {['Choice 1', 'Choice 2'][label]}"
        # text = question
        # text1 = sol1
        # text2 = sol2
    if dataset_name == "story-cloze":
        return f"Which choice makes the most sense? \nStory: {text}\nContinuation 1: {text1}\nContinuation 2: {text2} \nAnswer here: {['Continuation 1', 'Continuation 2'][label]}"
        # text = context
        # text1 = sentence_quiz1
        # text2 = sentence_quiz2

def test_formatting(data, dataset_name, n=100):
    """
    Given an encoder-decoder model, a list of data, computes the contrast hidden states on n random examples.
    Returns numpy arrays of shape (n, hidden_dim) for each candidate label, along with a boolean numpy array of shape (n,)
    with the ground truth labels
    
    This is deliberately simple so that it's easy to understand, rather than being optimized for efficiency
    """
    print("***** HI this func bein call rn")
    # setup
    all_neg_hs, all_pos_hs, all_gt_labels = [], [], []

    # loop
    for _ in tqdm(range(n)):
        # for simplicity, sample a random example until we find one that's a reasonable length
        # (most examples should be a reasonable length, so this is just to make sure)
        # while True:
        idx = np.random.randint(len(data))
        text, true_label = data[idx]["text"], data[idx]["label"]
        print(f"** Text: {true_label}")
        print(f"** True label: {true_label}")
        try:
            text1 = data[idx]["text1"]
        except:
            text1 = ""
        try:
            text2 = data[idx]["text2"]
        except:
            text2 = ""
        # the actual formatted input will be longer, so include a bit of a marign
        # if len(tokenizer(text + text1 + text2)) < 300: 
        #     # print("Skipped an example that was too long: " + text + text1 + text2)
        #     break

        # print(f"Number of tokens: {len(tokenizer(text + text1 + text2))}")

        for i, label in enumerate(label_dict.get(dataset_name)):
            if i != true_label:
                print(f"** False label: {i}")
                print(f"** True label: {true_label}")

                # get hidden states
                print("** Doing negative prompt now")
                neg_prompt = format_prompt(i, text, text1, text2, dataset_name = dataset_name)
                print(neg_prompt)

                print("** Doing positive prompt now")
                pos_prompt = format_prompt(true_label, text, text1, text2, dataset_name = dataset_name)
                print(pos_prompt)

# for dataset_name in dataset_names_singular:
#     test_formatting(datasets[dataset_name], dataset_name, n=1)

#%%

from transformers import AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer

# if you want to cache the model weights somewhere, you can specify that here
cache_dir = None

def load_model_helper(full_model_name):
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(full_model_name, cache_dir=cache_dir, torch_dtype = torch.half)
        model_type = "encoder_decoder"
    except:
        try:
            model = AutoModelForMaskedLM.from_pretrained(full_model_name, cache_dir=cache_dir, torch_dtype = torch.half)
            model_type = "encoder"
        except:
            model = AutoModelForCausalLM.from_pretrained(full_model_name, cache_dir=cache_dir, torch_dtype = torch.half)
            model_type = "decoder"
    tokenizer = AutoTokenizer.from_pretrained(full_model_name, cache_dir=cache_dir, torch_dtype = torch.half)
    model = model.half()
    return model, model_type, tokenizer

def load_model(model_name):
    if model_name == "deberta":
        full_model_name = "microsoft/deberta-v2-xxlarge"
        if models_downloaded == True:
            full_model_name = "deberta-v2-xxlarge"
        model, model_type, tokenizer = load_model_helper(full_model_name)
        model.to(device)
    elif model_name == "gpt-j":
        full_model_name = "EleutherAI/gpt-j-6B"
        if models_downloaded == True:
            full_model_name = "gpt-j-6B"
        model, model_type, tokenizer = load_model_helper(full_model_name)
        model.to(device)
    elif model_name == "t5":
        full_model_name = "t5-11b"
        if models_downloaded == True:
            full_model_name = "t5-11b"
        model, model_type, tokenizer = load_model_helper(full_model_name)
        model.to(device)
        # model.parallelize()  # T5 is big enough that we may need to run it on multiple GPUs
    elif model_name == "unifiedqa":
        full_model_name = "allenai/unifiedqa-t5-11b"
        if models_downloaded == True:
            full_model_name = "unifiedqa-t5-11b"
        model, model_type, tokenizer = load_model_helper(full_model_name)
        model.to(device)
    elif model_name == "T0pp":
        full_model_name = "bigscience/T0pp"
        if models_downloaded == True:
            full_model_name = "T0pp"
        model, model_type, tokenizer = load_model_helper(full_model_name)
        model.to(device)
    else:
        print(f"Not implemented! {model_name}")    
        return None, None, None
    print("Models loaded successfully!")
    return model, model_type, tokenizer


#%% Extract hidden states

def get_encoder_hidden_states(model, tokenizer, input_text, layer=-1):
    """
    Given an encoder model and some text, gets the encoder hidden states (in a given layer, by default the last) 
    on that input text (where the full text is given to the encoder).

    Returns a numpy array of shape (hidden_dim,)
    """
    # tokenize
    encoder_text_ids = tokenizer(input_text, truncation=True, return_tensors="pt").input_ids.to(model.device)

    # forward pass
    with torch.no_grad():
        output = model(encoder_text_ids, output_hidden_states=True)

    # get the appropriate hidden states
    hs_tuple = output["hidden_states"] # tupple of all hidden states at each layer of the encoder --> each layer is (batch_size, sequence_length, hidden_dim)
    
    hs = hs_tuple[layer][0, -1].detach().cpu().numpy() # at a given layer. Then, 0 refers to first and only input in the batch, while -1 refers to last token in sequence.

    return hs

def get_encoder_decoder_hidden_states(model, tokenizer, input_text, layer=-1):
    """
    Given an encoder-decoder model and some text, gets the encoder hidden states (in a given layer, by default the last) 
    on that input text (where the full text is given to the encoder).

    Returns a numpy array of shape (hidden_dim,)
    """
    # tokenize
    encoder_text_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    decoder_text_ids = tokenizer("", return_tensors="pt").input_ids.to(model.device)

    # forward pass
    with torch.no_grad():
        output = model(encoder_text_ids, decoder_input_ids=decoder_text_ids, output_hidden_states=True)

    # get the appropriate hidden states
    hs_tuple = output["encoder_hidden_states"]
    hs = hs_tuple[layer][0, -1].detach().cpu().numpy()

    return hs

def get_decoder_hidden_states(model, tokenizer, input_text, layer=-1):
    """
    Given a decoder model and some text, gets the hidden states (in a given layer, by default the last) on that input text

    Returns a numpy array of shape (hidden_dim,)
    """
    # tokenize (adding the EOS token this time)
    input_ids = tokenizer(input_text + tokenizer.eos_token, return_tensors="pt").input_ids.to(model.device)

    # forward pass
    with torch.no_grad():
        output = model(input_ids, output_hidden_states=True)

    # get the last layer, last token hidden states
    hs_tuple = output["hidden_states"]
    hs = hs_tuple[layer][0, -1].detach().cpu().numpy()

    return hs

def get_hidden_states(model, tokenizer, input_text, layer=-1, model_type="encoder"):
    fn = {"encoder": get_encoder_hidden_states, "encoder_decoder": get_encoder_decoder_hidden_states,
          "decoder": get_decoder_hidden_states}[model_type]

    return fn(model, tokenizer, input_text, layer=layer)

def get_hidden_states_many_examples(model, tokenizer, data, model_type, dataset_name, n=100, used_idx = []):
    """
    Given an encoder-decoder model, a list of data, computes the contrast hidden states on n random examples.
    Returns numpy arrays of shape (n, hidden_dim) for each candidate label, along with a boolean numpy array of shape (n,)
    with the ground truth labels
    
    This is deliberately simple so that it's easy to understand, rather than being optimized for efficiency
    """
    # setup
    model.eval()
    all_neg_hs, all_pos_hs, all_gt_labels = [], [], []

    # BALANCE
    neg_labels = 0
    pos_labels = 0
    neg_labels_limit = n//2
    print(f"neg_labels_limit: {neg_labels_limit}")
    pos_labels_limit = n-neg_labels_limit
    print(f"pos_labels_limit: {pos_labels_limit}")

    # NO REPEATS
    print(f"intial used_idx: {used_idx}")
    used_idxs = used_idx
    print(f"intial used_idxs: {used_idxs}")

    # loop
    for _ in tqdm(range(n)):
        # for simplicity, sample a random example until we find one that's a reasonable length
        # (most examples should be a reasonable length, so this is just to make sure)
        while True:
            idx = np.random.randint(len(data))
            if idx in used_idxs:
                continue
            text, true_label = data[idx]["text"], data[idx]["label"]
            try:
                text1 = data[idx]["text1"]
            except:
                text1 = ""
            try:
                text2 = data[idx]["text2"]
            except:
                text2 = ""
            # the actual formatted input will be longer, so include a bit of a marign
            # print("Example: " + text + text1 + text2)
            len_requirement = len(tokenizer(text + text1 + text2)) < 300 
            if not len_requirement:
                print("Skipped an example that was too long: " + len(tokenizer(text + text1 + text2)))
            if pos_labels < pos_labels_limit and true_label == 1 and len_requirement:
                used_idxs.append(idx)
                break
            elif neg_labels < neg_labels_limit and true_label == 0 and len_requirement:
                used_idxs.append(idx)
                break

        # print(f"Number of tokens: {len(tokenizer(text + text1 + text2))}")

        if true_label == 0:
            neg_labels += 1
        else:
            pos_labels += 1

        for i, label in enumerate(label_dict.get(dataset_name)):

            if i != true_label:

                # get hidden states
                neg_prompt = format_prompt(i, text, text1, text2, dataset_name = dataset_name)
                print("number of tokens in neg_prompt: " + str(len(tokenizer(neg_prompt))))
                if (str(len(tokenizer(neg_prompt))) > "512"):
                    print("[ERROR] PROMPT TOO LONG: " + neg_prompt)
                # print(neg_prompt)
                neg_hs = get_hidden_states(model, tokenizer, neg_prompt, model_type=model_type)

                pos_prompt = format_prompt(true_label, text, text1, text2, dataset_name = dataset_name)
                print("number of tokens in pos_prompt: " + str(len(tokenizer(neg_prompt))))
                # print(pos_prompt)
                pos_hs = get_hidden_states(model, tokenizer, pos_prompt, model_type=model_type)
                if (str(len(tokenizer(neg_prompt))) > "512"):
                    print("[ERROR] PROMPT TOO LONG: " + pos_prompt)

                # collect
                all_neg_hs.append(neg_hs)
                all_pos_hs.append(pos_hs)
                all_gt_labels.append(true_label)

    all_neg_hs = np.stack(all_neg_hs)
    all_pos_hs = np.stack(all_pos_hs)
    all_gt_labels = np.stack(all_gt_labels)

    return all_neg_hs, all_pos_hs, all_gt_labels, used_idxs

#%%

class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        return torch.sigmoid(o)

class CCS(object):
    def __init__(self, x0, x1, y, nepochs=1000, ntries=10, lr=1e-3, batch_size=-1, 
                 verbose=False, device=device, linear=True, weight_decay=0.01, var_normalize=False):
        # data
        self.var_normalize = var_normalize
        self.x0 = self.normalize(x0)
        self.x1 = self.normalize(x1)
        self.y = y
        self.d = self.x0.shape[-1]

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        # probe
        self.linear = linear
        self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)

        
    def initialize_probe(self):
        if self.linear:
            self.probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid())
        else:
            self.probe = MLPProbe(self.d)
        self.probe.to(self.device)    


    def normalize(self, x):
        """
        Mean-normalizes the data x (of shape (n, d))
        If self.var_normalize, also divides by the standard deviation
        """
        normalized_x = x - x.mean(axis=0, keepdims=True)
        if self.var_normalize:
            normalized_x /= normalized_x.std(axis=0, keepdims=True)

        return normalized_x

        
    def get_tensor_data(self):
        """
        Returns x0, x1 as appropriate tensors (rather than np arrays)
        """
        x0 = torch.tensor(self.x0, dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.x1, dtype=torch.float, requires_grad=False, device=self.device)
        return x0, x1
    
    def get_loss(self, p0, p1):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        informative_loss = (torch.min(p0, p1)**2).mean(0)
        consistent_loss = ((p0 - (1-p1))**2).mean(0)
        return informative_loss + consistent_loss

    def get_train_dir(self):
        """
        Returns the correct probe direction for "true" on the train data.
        """
        x0, x1 = self.get_tensor_data()
        with torch.no_grad():
            p0, p1 = self.probe(x0), self.probe(x1)
        avg_confidence = 0.5*(p0 + (1-p1))
        predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int)[:, 0]
        acc = (predictions == self.y).mean()
        if acc > 0.5:
            return False # don't turn
        else:
            return True # do turn

    def get_acc(self, x0_test, x1_test, y_test):
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        x0 = torch.tensor(self.normalize(x0_test), dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.normalize(x1_test), dtype=torch.float, requires_grad=False, device=self.device)
        with torch.no_grad():
            p0, p1 = self.best_probe(x0), self.best_probe(x1)
        avg_confidence = 0.5*(p0 + (1-p1))
        predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int)[:, 0]
        acc = (predictions == y_test).mean()
        if self.get_train_dir(): # Apply train_direction
            acc = 1 - acc
        # acc = max(acc, 1-acc)
        return acc
    
        
    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x0, x1 = self.get_tensor_data()
        permutation = torch.randperm(len(x0))
        x0, x1 = x0[permutation], x1[permutation]
        
        # set up optimizer
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        nbatches = len(x0) // batch_size

        # Start training (full batch)
        for epoch in range(self.nepochs):
            for j in range(nbatches):
                x0_batch = x0[j*batch_size:(j+1)*batch_size]
                x1_batch = x1[j*batch_size:(j+1)*batch_size]
            
                # probe
                p0, p1 = self.probe(x0_batch), self.probe(x1_batch)

                # get the corresponding loss
                loss = self.get_loss(p0, p1)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss.detach().cpu().item()
    
    def repeated_train(self):
        best_loss = np.inf
        for train_num in range(self.ntries):
            self.initialize_probe()
            loss = self.train()
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss

        return best_loss

#%% 

num_epochs = 350

model_names = ["deberta", "gpt-j"]
# model_names = ["deberta", "gpt-j", "t5", "unifiedqa", "T0pp"]

# 3D torch Tensor of results
# 0th dimension: models
# 1st dimension: train datasets
# 2nd dimension: test datasets
ccs_train = torch.zeros((len(model_names), len(dataset_names_singular), len(dataset_names_singular)))
probe_train = torch.zeros((len(model_names), len(dataset_names_singular), len(dataset_names_singular)))
ccs_test = torch.zeros((len(model_names), len(dataset_names_singular), len(dataset_names_singular)))
probe_test = torch.zeros((len(model_names), len(dataset_names_singular), len(dataset_names_singular)))

for i, model_name in enumerate(model_names):
    model, model_type, tokenizer = load_model(model_name)
    print(" **** Loaded model: ", model_name)

    for j, dataset_name_train in enumerate(dataset_names_singular):
        for k, dataset_name_test in enumerate(dataset_names_singular):

            print(" *** Training on ", dataset_name_train, " and testing on ", dataset_name_test)

            # dataset_name_train = "imdb"
            # dataset_name_test = "amazon_polarity"
            data_train = datasets[dataset_name_train]
            data_test = datasets[dataset_name_test]
            print("loaded data")

            print(f"Memory used, total: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
            if j != k:
                neg_hs_train, pos_hs_train, y_train, _ = get_hidden_states_many_examples(model, tokenizer, data_train, model_type, dataset_name_train, num_epochs, used_idx = [])
                print("got hidden states, train")
                print(f"Memory used, total: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
                neg_hs_test, pos_hs_test, y_test, _ = get_hidden_states_many_examples(model, tokenizer, data_test, model_type, dataset_name_test, num_epochs, used_idx = [])
                print("got hidden states, test")
                print(f"Memory used, total: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
            else:
                neg_hs_train, pos_hs_train, y_train, used_indx = get_hidden_states_many_examples(model, tokenizer, data_train, model_type, dataset_name_train, num_epochs, used_idx = [])
                neg_hs_test, pos_hs_test, y_test, used_indx = get_hidden_states_many_examples(model, tokenizer, data_train, model_type, dataset_name_train, num_epochs, used_indx)
                # If j = k, let's split training / testing results. without repetition, to ensur ethat the balance code in get_hidden_States_many_examples stays that way

            # for simplicity we can just take the difference between positive and negative hidden states
            # (concatenating also works fine)
            x_train = neg_hs_train - pos_hs_train
            x_test = neg_hs_test - pos_hs_test

            # Run a probe on neg and pos hidden states
            try:
                lr = LogisticRegression(class_weight="balanced")
                print("Shape of x_train: ", x_train.shape)
                print("Shape of y_train: ", y_train.shape)
                lr.fit(x_train, y_train)
                print("Logistic regression accuracy on transfer: {}".format(lr.score(x_test, y_test)))
                print("Logistic regression accuracy on own: {}".format(lr.score(x_train, y_train)))
                probe_train[i, j, k] = lr.score(x_train, y_train)
                probe_test[i, j, k] = lr.score(x_test, y_test)
            except:
                print("Logistic regression failed")

            # Train and run CCS without any labels
            try:
                ccs = CCS(neg_hs_train, pos_hs_train, y_train)
                ccs.repeated_train()
                ccs_acc_train = ccs.get_acc(neg_hs_train, pos_hs_train, y_train)
                ccs_acc_test = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
                print("CCS accuracy on transfer: {}".format(ccs_acc_test))
                print("CCS accuracy on own: {}".format(ccs_acc_train))
                ccs_train[i, j, k] = ccs_acc_train
                ccs_test[i, j, k] = ccs_acc_test
            except:
                print("CCS failed")
    # Save ccs_results tensor
    torch.save(ccs_test, f'ccs_test_{model_name}.pt')
    torch.save(ccs_train, f'ccs_train_{model_name}.pt')

    # Save probe_results tensor
    torch.save(probe_test, f'probe_test_{model_name}.pt')
    torch.save(probe_train, f'probe_train_{model_name}.pt')

    del model
    del tokenizer
    torch.cuda.empty_cache()

#%%

# model_name = "deberta"
# model, model_type, tokenizer = load_model(model_name)
# # del model

# #%%

# def train_test_one(dataset_name_train, dataset_name_test, num_epochs=350):
#     print(" *** Training on ", dataset_name_train, " and testing on ", dataset_name_test)

#     # dataset_name_train = "imdb"
#     # dataset_name_test = "amazon_polarity"
#     data_train = datasets[dataset_name_train]
#     data_test = datasets[dataset_name_test]

#     print("loaded data")

#     # if dataset_name_train != dataset_name_test:
#     neg_hs_train, pos_hs_train, y_train, _ = get_hidden_states_many_examples(model, tokenizer, data_train, model_type, dataset_name_train, num_epochs)
#     print("got train hidden state")
#     neg_hs_test, pos_hs_test, y_test, _ = get_hidden_states_many_examples(model, tokenizer, data_test, model_type, dataset_name_test, num_epochs)
#     print("got test hidden state")
#     # else:
#     #     neg_hs, pos_hs, y = get_hidden_states_many_examples(model, tokenizer, data_train, model_type, dataset_name_train, num_epochs*2)
#     #     # If j = k, let's split training / testing results:
#     #     n = len(y)
#     #     neg_hs_train, neg_hs_test = neg_hs[:n//2], neg_hs[n//2:]
#     #     pos_hs_train, pos_hs_test = pos_hs[:n//2], pos_hs[n//2:]
#     #     y_train, y_test = y[:n//2], y[n//2:]

#     print("loaded hidden states")

#     # for simplicity we can just take the difference between positive and negative hidden states
#     # (concatenating also works fine)
#     x_train = neg_hs_train - pos_hs_train
#     x_test = neg_hs_test - pos_hs_test

#     # Run a probe on neg and pos hidden states
#     # try:
#     lr = LogisticRegression(class_weight="balanced")
#     print("Shape of x_train: ", x_train.shape)
#     print("Shape of y_train: ", y_train.shape)
#     print(f"Y_train: {y_train}")
#     lr.fit(x_train, y_train)
#     print("Logistic regression accuracy on transfer: {}".format(lr.score(x_test, y_test)))
#     print("Logistic regression accuracy on own: {}".format(lr.score(x_train, y_train)))
#     # probe_results[i, j, k] = lr.score(x_test, y_test)
#     # except:
#     #     print("Logistic regression failed")

#     # Train and run CCS without any labels
#     ccs = CCS(neg_hs_train, pos_hs_train, y_train)
#     ccs.repeated_train()
#     ccs_acc_train = ccs.get_acc(neg_hs_train, pos_hs_train, y_train)
#     ccs_acc_test = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
#     print("CCS accuracy on transfer: {}".format(ccs_acc_test))
#     print("CCS accuracy on own: {}".format(ccs_acc_train))
#     # ccs_results[i, j, k] = ccs_acc_test

# train_test_one("amazon_polarity", "copa")

# #%%

# def train_test_one(dataset_name_train, dataset_name_test, num_epochs=350):
#     print(" *** Training on ", dataset_name_train, " and testing on ", dataset_name_test)

#     # dataset_name_train = "imdb"
#     # dataset_name_test = "amazon_polarity"
#     data_train = datasets[dataset_name_train]
#     data_test = datasets[dataset_name_test]

#     print("loaded data")

#     # if dataset_name_train != dataset_name_test:
#     neg_hs_train, pos_hs_train, y_train, _ = get_hidden_states_many_examples(model, tokenizer, data_train, model_type, dataset_name_train, num_epochs)
#     print("got train hidden state")
#     neg_hs_test, pos_hs_test, y_test, _ = get_hidden_states_many_examples(model, tokenizer, data_test, model_type, dataset_name_test, num_epochs)
#     print("got test hidden state")
#     # else:
#     #     neg_hs, pos_hs, y = get_hidden_states_many_examples(model, tokenizer, data_train, model_type, dataset_name_train, num_epochs*2)
#     #     # If j = k, let's split training / testing results:
#     #     n = len(y)
#     #     neg_hs_train, neg_hs_test = neg_hs[:n//2], neg_hs[n//2:]
#     #     pos_hs_train, pos_hs_test = pos_hs[:n//2], pos_hs[n//2:]
#     #     y_train, y_test = y[:n//2], y[n//2:]

#     print("loaded hidden states")

#     # for simplicity we can just take the difference between positive and negative hidden states
#     # (concatenating also works fine)
#     x_train = neg_hs_train - pos_hs_train
#     x_test = neg_hs_test - pos_hs_test

#     # Run a probe on neg and pos hidden states
#     # try:
#     lr = LogisticRegression(class_weight="balanced")
#     print("Shape of x_train: ", x_train.shape)
#     print("Shape of y_train: ", y_train.shape)
#     print(f"Y_train: {y_train}")
#     lr.fit(x_train, y_train)
#     print("Logistic regression accuracy on transfer: {}".format(lr.score(x_test, y_test)))
#     print("Logistic regression accuracy on own: {}".format(lr.score(x_train, y_train)))
#     # probe_results[i, j, k] = lr.score(x_test, y_test)
#     # except:
#     #     print("Logistic regression failed")

#     # Train and run CCS without any labels
#     ccs = CCS(neg_hs_train, pos_hs_train, y_train)
#     ccs.repeated_train()
#     ccs_acc_train = ccs.get_acc(neg_hs_train, pos_hs_train, y_train)
#     ccs_acc_test = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
#     print("CCS accuracy on transfer: {}".format(ccs_acc_test))
#     print("CCS accuracy on own: {}".format(ccs_acc_train))
#     # ccs_results[i, j, k] = ccs_acc_test

# train_test_one("amazon_polarity", "copa")

# #%%

# # def train_test_one(dataset_name_train, dataset_name_test, num_epochs=350):
# #     print(" *** Training on ", dataset_name_train, " and testing on ", dataset_name_test)

# #     # dataset_name_train = "imdb"
# #     # dataset_name_test = "amazon_polarity"
# #     data_train = datasets[dataset_name_train]
# #     data_test = datasets[dataset_name_test]
# #     print("loaded data")

# #     print(f"Memory used, total: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
# #     if dataset_name_train != dataset_name_test:
# #         neg_hs_train, pos_hs_train, y_train, _ = get_hidden_states_many_examples(model, tokenizer, data_train, model_type, dataset_name_train, num_epochs)
# #         print("got hidden states, train")
# #         print(f"Memory used, total: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
# #         neg_hs_test, pos_hs_test, y_test, _ = get_hidden_states_many_examples(model, tokenizer, data_test, model_type, dataset_name_test, num_epochs)
# #         print("got hidden states, test")
# #         print(f"Memory used, total: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
# #     else:
# #         neg_hs_train, pos_hs_train, y_train, used_indx = get_hidden_states_many_examples(model, tokenizer, data_train, model_type, dataset_name_train, num_epochs)
# #         neg_hs_test, pos_hs_test, y_test, used_indx = get_hidden_states_many_examples(model, tokenizer, data_train, model_type, dataset_name_train, num_epochs, used_indx)
# #         # If j = k, let's split training / testing results. without repetition, to ensur ethat the balance code in get_hidden_States_many_examples stays that way

# #     # for simplicity we can just take the difference between positive and negative hidden states
# #     # (concatenating also works fine)
# #     x_train = neg_hs_train - pos_hs_train
# #     x_test = neg_hs_test - pos_hs_test

# #     # Run a probe on neg and pos hidden states
# #     try:
# #         lr = LogisticRegression(class_weight="balanced")
# #         print("Shape of x_train: ", x_train.shape)
# #         print("Shape of y_train: ", y_train.shape)
# #         lr.fit(x_train, y_train)
# #         print("Logistic regression accuracy on transfer: {}".format(lr.score(x_test, y_test)))
# #         print("Logistic regression accuracy on own: {}".format(lr.score(x_train, y_train)))
# #         probe_train[i, j, k] = lr.score(x_train, y_train)
# #         probe_test[i, j, k] = lr.score(x_test, y_test)
# #     except:
# #         print("Logistic regression failed")

# #     # Train and run CCS without any labels
# #     try:
# #         ccs = CCS(neg_hs_train, pos_hs_train, y_train)
# #         ccs.repeated_train()
# #         ccs_acc_train = ccs.get_acc(neg_hs_train, pos_hs_train, y_train)
# #         ccs_acc_test = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
# #         print("CCS accuracy on transfer: {}".format(ccs_acc_test))
# #         print("CCS accuracy on own: {}".format(ccs_acc_train))
# #         ccs_train[i, j, k] = ccs_acc_train
# #         ccs_test[i, j, k] = ccs_acc_test
# #     except:
# #         print("CCS failed")

# # train_test_one("amazon_polarity", "copa")
# # #%%
#%%