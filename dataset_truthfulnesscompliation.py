#%%
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

#%% Load datasets
tqa = datasets.load_dataset("truthful_qa", "generation")["validation"]
cfact = datasets.load_dataset("NeelNanda/counterfact-tracing")["train"]
f1train_ = datasets.load_dataset("fever", "v1.0")["labelled_dev"] # has duplication
f1train = drop_dup_claim(f1train_)
f1dev_ = datasets.load_dataset("fever", "v1.0")["train"]
f1dev = drop_dup_claim(f1dev_)
f2_ = datasets.load_dataset("fever", "v2.0")["validation"]
f2 = drop_dup_claim(f2_)
bqtrain = datasets.load_dataset("boolq")["train"]
bqtest = datasets.load_dataset("boolq")["validation"]

claims = []
labels = []
explanation = []
common_knowledge_label = []
dataset = []

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
        claims.append(f"{question} {correct_answer}")
        labels.append(1)
        explanation.append(None)
        common_knowledge_label.append(None)
        dataset.append("TruthfulQA")
        
    # Append the question to each incorrect answer, add it to the claims list and add label '0'
    for incorrect_answer in incorrect_answers:
        claims.append(f"{question} {incorrect_answer}")
        labels.append(0)
        explanation.append(None)
        common_knowledge_label.append(None)
        dataset.append("TruthfulQA")

# Format CFact 
for i in tqdm(range(len(cfact))):
    row = cfact[i]
    
    prompt = row['prompt']
    target_true = row['target_true']
    target_false = row['target_false']
    
    # Concatenate the prompt with target_true, add to the claims list and add label '1'
    claims.append(f"{prompt}{target_true}")
    labels.append(1)
    explanation.append(None)
    common_knowledge_label.append(None)
    dataset.append("CounterFact")
    
    # Concatenate the prompt with target_false, add to the claims list and add label '0'
    claims.append(f"{prompt}{target_false}")
    labels.append(0)
    explanation.append(None)
    common_knowledge_label.append(None)
    dataset.append("CounterFact")


# Format FEVERv1 
for i in tqdm(range(len(f1dev))):
    row = f1dev[i]
    
    claim = row['claim']
    label = row['label']
    
    if label == "SUPPORTS":
        claims.append(claim)
        labels.append(1)
        explanation.append(None)
        common_knowledge_label.append(None)
        dataset.append("fever_v1.0_labelleddev")
    elif label == "REFUTES":
        claims.append(claim)
        labels.append(0)
        explanation.append(None)
        common_knowledge_label.append(None)
        dataset.append("fever_v1.0_labelleddev")
    else:
        continue

# Format FEVERv1 
for i in tqdm(range(len(f1train))):
    row = f1train[i]
    
    claim = row['claim']
    label = row['label']
    
    if label == "SUPPORTS":
        claims.append(claim)
        labels.append(1)
        explanation.append(None)
        common_knowledge_label.append(None)
        dataset.append("fever_v1.0_train")
    elif label == "REFUTES":
        claims.append(claim)
        labels.append(0)
        explanation.append(None)
        common_knowledge_label.append(None)
        dataset.append("fever_v1.0_train")
    else:
        continue

# Format FEVERv2
for i in tqdm(range(len(f2))):
    row = f2[i]
    
    claim = row['claim']
    label = row['label']
    
    if label == "SUPPORTS":
        claims.append(claim)
        labels.append(1)
        explanation.append(None)
        common_knowledge_label.append(None)
        dataset.append("fever_v2.0")
    elif label == "REFUTES":
        claims.append(claim)
        labels.append(0)
        explanation.append(None)
        common_knowledge_label.append(None)
        dataset.append("fever_v2.0")
    else:
        continue

# Format bq_train
for i in tqdm(range(len(bqtrain))):
    row = bqtrain[i]
    
    question = row['question']
    answer = row['answer']
    passage = row['passage']
    
    if label == "SUPPORTS":
        claims.append(claim)
        labels.append(1)
        explanation.append(passage)
        common_knowledge_label.append(None)
        dataset.append("boolq_train")
    elif label == "REFUTES":
        claims.append(claim)
        labels.append(0)
        explanation.append(passage)
        common_knowledge_label.append(None)
        dataset.append("boolq_train")
    else:
        continue

# Format bqtest
for i in tqdm(range(len(bqtest))):
    row = bqtest[i]
    
    question = row['question']
    answer = row['answer']
    passage = row['passage']
    
    if label == "SUPPORTS":
        claims.append(claim)
        labels.append(1)
        explanation.append(passage)
        common_knowledge_label.append(None)
        dataset.append("boolq_test")
    elif label == "REFUTES":
        claims.append(claim)
        labels.append(0)
        explanation.append(passage)
        common_knowledge_label.append(None)
        dataset.append("boolq_test")
    else:
        continue

# Format CREAK
for file in ['train.json', 'dev.json', 'contrast_set.json']:
    # Open the file
    with jsonlines.open(f'creak/{file}') as reader:
        # Go through each row
        for obj in reader:
            # Append the data to the lists
            claims.append(obj['sentence'])
            if obj['label'] == "true":
                labels.append(1)
            else:
                labels.append(0)
            explanation.append(obj['explanation'])
            common_knowledge_label.append(None)
            dataset.append("creak_" + file.split('.')[0])

# Format Common Claim
df = pd.read_csv('common_claim.csv')  # replace 'file.csv' with the path to your CSV file
claims.extend(df['examples'].tolist())
label_cc_list = [1 if label == True else 0 for label in df['label'].tolist()]
labels.extend(label_cc_list)
explanation.extend([None] * len(df))
dataset.extend(["CommonClaim"] * len(df))
common_knowledge_label.extend(df['agreement'].tolist())

#%%
# Now we have two lists 'claims' and 'labels' that we can use to create the DataFrame
df = pd.DataFrame({'claim': claims, 'label': labels, 'explanation': explanation, 'common_knowledge_label': common_knowledge_label, 'origin_dataset': dataset})
# %%

info = DatasetInfo(
    description="Many truthfulness datasets, compiled",
    citation="Ren & Campbell, 2023",
    homepage="huggingface.com",
    license="Apache-2.0",
)
dataset = Dataset.from_pandas(df, info=info)
dataset_dict = DatasetDict({
    'train': dataset
})

# # %%
# dataset_dict.save_to_disk("./truthfulness")
dataset.push_to_hub("notrichardren/truthfulness")
# # %%

from datasets import load_from_disk, load_dataset

# dataset = load_from_disk('truthfulness')
dataset = load_dataset("notrichardren/truthfulness")
# #
