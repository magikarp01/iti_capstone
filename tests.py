#%%

from utils.dataset_utils import CCS_Dataset
from utils.probing_utils import ModelActs
from utils.model_utils import vicuna_7b
from datasets import load_dataset, Dataset

#%%
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

#%%

model = vicuna_7b()
dataset = load_dataset("imdb")["train"]

#%%
ccs_data = CCS_Dataset(label_dict, format_prompt, dataset, model.tokenizer)
probing_utils = ModelActs(model, ccs_data)

# %%

probing_utils.CCS_train(6, 2)

# %%

