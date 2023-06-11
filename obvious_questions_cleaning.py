#%%
import pandas as pd
import re

#%%
# Let's say df is your dataframe and 'text' is the column with the text
df = pd.read_csv('obvious_questions_uncleaned.csv', header = None)

df.columns = ['category', 'text']

def split_row(row):
    question = None
    correct_answer = None
    wrong_answer = None

    q_match = re.search(r'(Question:\s|Obvious Question: |Question \d+:\s|Q\d+:\s)?(.*?)(?= Answer:|Correct Answer:| Correct answer:|Correct answer:|Obvious Answer:)', row)
    if q_match:
        question = q_match.group(2).strip()

    c_a_match = re.search(r'(Correct Answer:| Correct answer:|Correct answer:|Obvious Answer:|Answer:)\s?(.*?)(?= Wrong Answer:| Wrong answer:|Incorrect Answer:)', row)
    if c_a_match:
        correct_answer = c_a_match.group(2).strip()

    w_a_match = re.search(r'(Wrong Answer:| Wrong answer:|Incorrect Answer:)\s?(.*)', row)
    if w_a_match:
        wrong_answer = w_a_match.group(2).strip()

    return pd.Series({
        'Question': question,
        'Correct Answer': correct_answer,
        'Wrong Answer': wrong_answer
    })

new_df = df['text'].apply(split_row)
# %%

df = pd.concat([df['category'], new_df], axis=1)
# %%

df.to_csv('obvious_questions_clean.csv', index=False)