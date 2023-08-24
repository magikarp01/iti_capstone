#%%

from datasets import load_dataset

# features: ['claim', 'label', 'dataset', 'qa_type', 'ind']

misuse = load_dataset("notrichardren/refuse-to-answer-prompts")
decept = load_dataset("notrichardren/deception-evals")
gpt = load_dataset("notrichardren/gpt_generated_10k")

#%%

misuse
misuse = misuse.rename_column("Topic", "dataset")
misuse = misuse.rename_column("Question", "claim")
misuse = misuse.rename_column("Label", "label")

def add_indices(example, idx):
    example["ind"] = idx
    return example
misuse =misuse.map(add_indices, with_indices=True)

def add_qatype(example):
    example["qa_type"] = 0
    return example
misuse = misuse.map(add_qatype)

# QA type 0

# %%
decept
decept = decept.rename_column("Question", "claim")
decept = decept.rename_column("Label", "label")

def add_indices(example, idx):
    example["ind"] = idx
    return example
decept = decept.map(add_indices, with_indices=True)

def add_qatype(example):
    example["qa_type"] = 2
    return example
decept = decept.map(add_qatype)


# %%
gpt

from datasets import concatenate_datasets

gpt = gpt.remove_columns(["Unnamed: 0", '__index_level_0__'])
gpt = gpt.rename_column("Question", "claim")
gpt = gpt.rename_column("Topic", "dataset")
gpt = gpt.rename_column("Correct", "label")

def add_qatype(example):
    example["qa_type"] = 0
    return example
gpt = gpt.map(add_qatype)
# QA type 0

# Make it all train

combined_train = concatenate_datasets([gpt['train'], gpt['test']])
# remove the other ones

del gpt['train']
del gpt['test']
gpt["train"] = combined_train

# %%
