#%%
# Import, setup, helper functions
import datasets
from datasets import Dataset, DatasetDict, DatasetInfo
import pandas as pd
import jsonlines
from tqdm import tqdm

def drop_dup_claim(dataset):
    # Convert to pandas DataFrame
    df = dataset.to_pandas()

    # Remove duplicates
    df = df.drop_duplicates(subset='claim')

    # Convert back to HuggingFace dataset
    new_dataset = Dataset.from_pandas(df)
    return new_dataset

# Data types
claim = []
label = []
dataset = [] # origin dataset
qa_type = [] # 0 for statement/answer only, 1 for q&a, 2 for question only
explanation = [] # explanation, if exists

# Variables used for splitting
expl_passage_dependent = [] # do you need the passage/explanation to answer the question? 0 for no, 1 for yes
hq = [] # is the dataset high-quality? 0 for no, 1 for yes

#%%

# TruthfulQA
tqa = datasets.load_dataset("truthful_qa", "generation")["validation"] # Generation and MC are the same

# Format TQA
for i in tqdm(range(len(tqa))):
    row = tqa[i]
    
    # Get the question
    question = row['question']
    
    # Get the sequences of correct and incorrect answers
    correct_answers = row['correct_answers']
    incorrect_answers = row['incorrect_answers']
    
    # Append the question to each correct answer, add it to the claims list and add label '1'
    for correct_answer in correct_answers:
        claim.append(f"{question} {correct_answer}")
        label.append(1)
        dataset.append("truthfulqa")
        qa_type.append(1)
        explanation.append(None)
        hq.append(1)
        expl_passage_dependent.append(0)
        
    # Append the question to each incorrect answer, add it to the claims list and add label '0'
    for incorrect_answer in incorrect_answers:
        claim.append(f"{question} {incorrect_answer}")
        label.append(0)
        dataset.append("truthfulqa")
        qa_type.append(1)
        explanation.append(None)
        hq.append(1)
        expl_passage_dependent.append(0)


#%%

# FEVERv1.0
f1train_ = datasets.load_dataset("fever", "v1.0")["labelled_dev"] # has duplication
f1train = drop_dup_claim(f1train_)
f1dev_ = datasets.load_dataset("fever", "v1.0")["train"] # has duplication
f1dev = drop_dup_claim(f1dev_)

# Format FEVERv1.0
for _dataset in [f1train, f1dev]:
    for i in tqdm(range(len(_dataset))):
        row = _dataset[i]
        
        _claim = row['claim']
        _label = row['label']
        
        if _label == "SUPPORTS":
            claim.append(_claim)
            label.append(1)
            dataset.append("fever_v1.0")
            qa_type.append(0)
            explanation.append(None)
            hq.append(0)
            expl_passage_dependent.append(0)
        elif _label == "REFUTES":
            claim.append(_claim)
            label.append(0)
            dataset.append("fever_v1.0")
            qa_type.append(0)
            explanation.append(None)
            hq.append(0)
            expl_passage_dependent.append(0)
        else:
            continue

#%%

# FEVERv2.0
f2_ = datasets.load_dataset("fever", "v2.0")["validation"]
f2 = drop_dup_claim(f2_)

# Format FEVERv2
for i in tqdm(range(len(f2))):
    row = f2[i]
    
    _claim = row['claim']
    _label = row['label']
    
    if _label == "SUPPORTS":
        claim.append(_claim)
        label.append(1)
        dataset.append("fever_v2.0")
        qa_type.append(0)
        explanation.append(None)
        hq.append(0)
        expl_passage_dependent.append(0)
    elif _label == "REFUTES":
        claim.append(_claim)
        label.append(0)
        dataset.append("fever_v2.0")
        qa_type.append(0)
        explanation.append(None)
        hq.append(0)
        expl_passage_dependent.append(0)
    else:
        continue

# %%
# BoolQ
bqtrain = datasets.load_dataset("boolq")["train"]
bqtest = datasets.load_dataset("boolq")["validation"]

# Format BoolQ

for _dataset in [bqtrain, bqtest]:
    for i in tqdm(range(len(_dataset))):
        row = _dataset[i]
        
        question = row['question']
        answer = row['answer']
        passage = row['passage']
        
        if answer == True:
            claim.append(question)
            label.append(1)
            dataset.append("boolq")
            qa_type.append(2)
            explanation.append(passage)
            hq.append(0)
            expl_passage_dependent.append(1)
        elif answer == False:
            claim.append(question)
            label.append(0)
            dataset.append("boolq")
            qa_type.append(2)
            explanation.append(passage)
            hq.append(0)
            expl_passage_dependent.append(1)
        else:
            continue

#%%
# CommonClaim
cc = pd.read_csv('other-datasets/common_claim.csv')  # replace 'file.csv' with the path to your CSV file

# Format CommonClaim
claim.extend(cc['examples'].tolist())
label.extend([1 if label == "True" else 0 for label in cc['label'].tolist()])
dataset.extend(["commonclaim"] * len(cc))
qa_type.extend([0]*len(cc))
explanation.extend([None]*len(cc))
hq.extend([1]*len(cc))
expl_passage_dependent.extend([0]*len(cc))

#%%

# Format CREAK
for file in ['train.json', 'dev.json']:
    # Open the file
    with jsonlines.open(f'other-datasets/creak/{file}') as reader:
        # Go through each row
        for obj in reader:
            # Append the data to the lists
            claim.append(obj['sentence'])
            if obj['label'] == "true":
                label.append(1)
            else:
                label.append(0)
            dataset.append("creak")
            qa_type.append(0)
            explanation.append(obj['explanation'])
            hq.append(1)
            expl_passage_dependent.append(0)

for file in ['contrast_set.json']:
    # Open the file
    with jsonlines.open(f'other-datasets/creak/{file}') as reader:
        # Go through each row
        for obj in reader:
            # Append the data to the lists
            claim.append(obj['sentence'])
            if obj['label'] == "true":
                label.append(1)
            else:
                label.append(0)
            dataset.append("creak")
            qa_type.append(0)
            explanation.append(None)
            hq.append(1)
            expl_passage_dependent.append(0)

# %%

# Counterfact
cfact = datasets.load_dataset("NeelNanda/counterfact-tracing")["train"]

# Format CounterFact

for i in tqdm(range(len(cfact))):
    row = cfact[i]
    
    prompt = row['prompt']
    target_true = row['target_true']
    target_false = row['target_false']
    
    # Concatenate the prompt with target_true, add to the claims list and add label '1'
    claim.append(f"{prompt}{target_true}")
    label.append(1)
    dataset.append("counterfact")
    qa_type.append(0)
    explanation.append(None)
    hq.append(0)
    expl_passage_dependent.append(0)

    # Concatenate the prompt with target_true, add to the claims list and add label '1'
    claim.append(f"{prompt}{target_false}")
    label.append(0)
    dataset.append("counterfact")
    qa_type.append(0)
    explanation.append(None)
    hq.append(0)
    expl_passage_dependent.append(0)
    
#%%

# SciQ
sciq_train = datasets.load_dataset("sciq")["train"]
sciq_val = datasets.load_dataset("sciq")["validation"]
sciq_test = datasets.load_dataset("sciq")["test"]

# Format SciQ
for _dataset in [sciq_train, sciq_val, sciq_test]:
    for i in tqdm(range(len(_dataset))):
        row = _dataset[i]
        
        question = row['question']
        distractor3 = row['distractor3']
        distractor2 = row['distractor2']
        distractor1 = row['distractor1']
        correct_answer = row['correct_answer']
        support = row['support']
        
        claim.append(f"{question} {correct_answer}")
        label.append(1)
        dataset.append("sciq")
        qa_type.append(1)
        explanation.append(support)
        hq.append(1)
        expl_passage_dependent.append(0)

        for [distractor, _label] in [[distractor1, 0], [distractor2, 0], [distractor3, 0]]:
            claim.append(f"{question} {distractor}")
            label.append(_label)
            dataset.append("sciq")
            qa_type.append(1)
            explanation.append(support)
            hq.append(1)
            expl_passage_dependent.append(0)

#%%

# Azaria & Mitchell
animals = pd.read_csv('other-datasets/azaria-mitchell/animals_true_false.csv')
capitals = pd.read_csv('other-datasets/azaria-mitchell/capitals_true_false.csv')
cities = pd.read_csv('other-datasets/azaria-mitchell/cities_true_false.csv')
companies = pd.read_csv('other-datasets/azaria-mitchell/companies_true_false.csv')
elements = pd.read_csv('other-datasets/azaria-mitchell/elements_true_false.csv')
inventions = pd.read_csv('other-datasets/azaria-mitchell/inventions_true_false.csv')
facts = pd.read_csv('other-datasets/azaria-mitchell/facts_true_false.csv')

for df, df_name in [[animals, "animals"], [capitals, "capitals"], [cities, "cities"], [companies, "companies"], [elements, "elements"], [inventions, "inventions"], [facts, "facts"]]:
    for index, row in df.iterrows():
        statement, _label = row['statement'], row['label']
        claim.append(statement)
        label.append(_label)
        dataset.append("azaria_mitchell_" + df_name)
        qa_type.append(0)
        explanation.append(None)
        hq.append(1)
        expl_passage_dependent.append(0)

#%%
df = pd.DataFrame({'claim': claim, 'label': label, 'dataset': dataset, 'qa_type': qa_type, 'explanation': explanation, 'hq': hq, 'expl_passage_dependent': expl_passage_dependent})
df

#%%
from sklearn.model_selection import train_test_split

def _reindex(df):
    df.reset_index(drop = True)
    df['ind'] = range(0, len(df))
    print(df)
    return df

def tts_and_reindex(df, test_size=0.2, random_state=42):
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train, test = _reindex(train), _reindex(test)
    all = _reindex(df)
    return train, test, all

def drop_duplicates_and_shuffle(df, random_state = 42):
    df = df.drop_duplicates()
    
    df = df.sample(frac=1, random_state = random_state).reset_index(drop = True)
    return df

def embeddings_ood_tts(df):
    """
    Split the dataframe into train and test sets by taking the BERT embedding of each claim and splitting based on a random cluster.
    """
    ...

# %%

# Upload the truthfulness_all dataset.

df = pd.DataFrame({'claim': claim, 'label': label, 'dataset': dataset, 'qa_type': qa_type})
df = drop_duplicates_and_shuffle(df)
# Count the amount of 'dataset' = truthfulqa, fever_v1.0, fever_v2.0, boolq, commonclaim, creak, counterfact, sciq, azaria_mitchell_animals, azaria_mitchell_capitals, azaria_mitchell_cities, azaria_mitchell_companies, azaria_mitchell_elements, azaria_mitchell_inventions, azaria_mitchell_facts in notrichardren/truthfulness_all.
counts = df['dataset'].value_counts()
print(counts)
counts = df['qa_type'].value_counts()
print(counts)
train_df, test_df, all_df = tts_and_reindex(df)

all = Dataset.from_pandas(all_df, preserve_index = False)
train = Dataset.from_pandas(train_df, preserve_index = False)
test = Dataset.from_pandas(test_df, preserve_index = False)
dataset_dict = DatasetDict({
    'all': all,
    'train': train,
    'test': test
})

dataset_dict.push_to_hub("notrichardren/truthfulness_all")

#%%

# Upload the truthfulness_high_quality dataset.

df = pd.DataFrame({'claim': claim, 'label': label, 'dataset': dataset, 'qa_type': qa_type, 'hq': hq})
df = df[df['hq'] == 1]
df = drop_duplicates_and_shuffle(df)
train_df, test_df, all_df = tts_and_reindex(df)
counts = df['dataset'].value_counts()
print(counts)
counts = df['qa_type'].value_counts()
print(counts)

all = Dataset.from_pandas(all_df, preserve_index = False).remove_columns(['hq'])
train = Dataset.from_pandas(train_df, preserve_index = False).remove_columns(['hq'])
test = Dataset.from_pandas(test_df, preserve_index = False).remove_columns(['hq'])
dataset_dict = DatasetDict({
    'all': all,
    'train': train,
    'test': test
})

dataset_dict.push_to_hub("notrichardren/truthfulness_high_quality")

#%%

# Upload the truthfulness_passage dataset. First, take all the dataframe where explanation is not None. Then, split it based on whether expl_passage_dependent is 1 or 0.

df = pd.DataFrame({'claim': claim, 'label': label, 'dataset': dataset, 'qa_type': qa_type, 'explanation': explanation, 'expl_passage_dependent': expl_passage_dependent})
df = df[df['explanation'].notna()]
df = drop_duplicates_and_shuffle(df)

# Split based on whether expl_passage_dependent is 1 or 0
df_1 = df[df['expl_passage_dependent'] == 1]
counts = df['dataset'].value_counts()
print(counts)
counts = df['qa_type'].value_counts()
print(counts)
df_0 = df[df['expl_passage_dependent'] == 0]
counts = df['dataset'].value_counts()
print(counts)
counts = df['qa_type'].value_counts()
print(counts)

df_1 = _reindex(df_1)
df_0 = _reindex(df_0)

passage_dependent = Dataset.from_pandas(df_1, preserve_index = False).remove_columns(['expl_passage_dependent'])
passage_independent = Dataset.from_pandas(df_0, preserve_index = False).remove_columns(['expl_passage_dependent'])

dataset_dict = DatasetDict({
    'passage_dependent': passage_dependent,
    'passage_independent': passage_independent
})

dataset_dict.push_to_hub("notrichardren/truthfulness_explanation")

# %%

# Upload the fever_overfit_test dataset.

df = pd.DataFrame({'claim': claim, 'label': label, 'dataset': dataset, 'qa_type': qa_type})
df1 = df[df['dataset'] == 'fever_v1.0']
df2 = df[df['dataset'] == "fever_v2.0"]
df = pd.concat([df1, df2])
df = drop_duplicates_and_shuffle(df)
# Count the amount of 'dataset' = truthfulqa, fever_v1.0, fever_v2.0, boolq, commonclaim, creak, counterfact, sciq, azaria_mitchell_animals, azaria_mitchell_capitals, azaria_mitchell_cities, azaria_mitchell_companies, azaria_mitchell_elements, azaria_mitchell_inventions, azaria_mitchell_facts in notrichardren/truthfulness_all.
counts = df['dataset'].value_counts()
print(counts)
counts = df['qa_type'].value_counts()
print(counts)
train_df, test_df, all_df = tts_and_reindex(df)

all = Dataset.from_pandas(all_df, preserve_index = False)
train = Dataset.from_pandas(train_df, preserve_index = False)
test = Dataset.from_pandas(test_df, preserve_index = False)
dataset_dict = DatasetDict({
    'all': all,
    'train': train,
    'test': test
})

dataset_dict.push_to_hub("notrichardren/fever_overfit_test")
# %%
