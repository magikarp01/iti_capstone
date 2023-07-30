
#%%
from datasets import load_dataset, DatasetInfo, DatasetDict
from datasets import Dataset
from datasets import load_dataset, concatenate_datasets, DatasetDict
import numpy as np

import pandas as pd
import numpy as np

from datasets import load_dataset
from tqdm import tqdm
import random

#%%
dataset = load_dataset("notrichardren/truthfulness")

np.random.seed(0)  # for reproducibility
indices = np.random.choice(range(len(dataset['train'])), size=20000, replace=False)
subset = dataset['train'].select(indices)
train_test_split = subset.train_test_split(test_size=0.2)

train_test_split.push_to_hub(f"notrichardren/truthfulness_20k")
# %%

hs_tf = load_dataset('notrichardren/hs_tf')
misconceptions_tf = load_dataset('notrichardren/misconceptions_tf')
ms_tf = load_dataset('notrichardren/ms_tf')
elem_tf = load_dataset('notrichardren/elem_tf')
kindergarten_tf = load_dataset('notrichardren/kindergarten_tf')
combined_dataset = concatenate_datasets([hs_tf['train'], misconceptions_tf['train'], ms_tf['train'], elem_tf['train'], kindergarten_tf['train']])

hs_tf = hs_tf.map(lambda x: {'original_dataset': 'hs_tf'}, remove_columns=['original_dataset'] if 'original_dataset' in hs_tf.column_names else [])
misconceptions_tf = misconceptions_tf.map(lambda x: {'original_dataset': 'misconceptions_tf'}, remove_columns=['original_dataset'] if 'original_dataset' in misconceptions_tf.column_names else [])
ms_tf = ms_tf.map(lambda x: {'original_dataset': 'ms_tf'}, remove_columns=['original_dataset'] if 'original_dataset' in ms_tf.column_names else [])
elem_tf = elem_tf.map(lambda x: {'original_dataset': 'elem_tf'}, remove_columns=['original_dataset'] if 'original_dataset' in elem_tf.column_names else [])
kindergarten_tf = kindergarten_tf.map(lambda x: {'original_dataset': 'kindergarten_tf'}, remove_columns=['original_dataset'] if 'original_dataset' in kindergarten_tf.column_names else [])

combined_dataset = concatenate_datasets([hs_tf['train'], misconceptions_tf['train'], ms_tf['train'], elem_tf['train'], kindergarten_tf['train']])
train_test_split = combined_dataset.train_test_split(test_size=0.2)

train_test_split.push_to_hub(f"notrichardren/gpt_generated_10k")
# %%
