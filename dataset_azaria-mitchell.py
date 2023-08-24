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

#%%

# Azaria & Mitchell
animals = pd.read_csv('other-datasets/azaria-mitchell/animals_true_false.csv')
capitals = pd.read_csv('other-datasets/azaria-mitchell/capitals_true_false.csv')
cities = pd.read_csv('other-datasets/azaria-mitchell/cities_true_false.csv')
companies = pd.read_csv('other-datasets/azaria-mitchell/companies_true_false.csv')
conj_neg_companies = pd.read_csv('other-datasets/azaria-mitchell/conj_neg_companies_true_false.csv')
conj_neg_facts = pd.read_csv('other-datasets/azaria-mitchell/conj_neg_facts_true_false.csv')
elements = pd.read_csv('other-datasets/azaria-mitchell/elements_true_false.csv')
facts = pd.read_csv('other-datasets/azaria-mitchell/facts_true_false.csv')
generated = pd.read_csv('other-datasets/azaria-mitchell/generated_true_false.csv')
inventions = pd.read_csv('other-datasets/azaria-mitchell/inventions_true_false.csv')
neg_companies = pd.read_csv('other-datasets/azaria-mitchell/neg_companies_true_false.csv')
neg_facts = pd.read_csv('other-datasets/azaria-mitchell/neg_facts_true_false.csv')

for df, df_name in [[animals, "animals"], [capitals, "capitals"], [cities, "cities"], [companies, "companies"], [elements, "elements"], [inventions, "inventions"], [facts, "facts"], [generated, "generated"], [neg_companies, "neg_companies"], [neg_facts, "neg_facts"], [conj_neg_companies, "conj_neg_companies"], [conj_neg_facts, "conj_neg_facts"]]:
    for index, row in df.iterrows():
        statement, _label = row['statement'], row['label']
        claim.append(statement)
        label.append(_label)
        dataset.append(df_name)
        qa_type.append(0)

#%%
df = pd.DataFrame({'claim': claim, 'label': label, 'dataset': dataset, 'qa_type': qa_type})

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
    'combined': all,
    'train': train,
    'test': test
})

dataset_dict.push_to_hub("notrichardren/azaria-mitchell")

#%%