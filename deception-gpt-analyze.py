#%%

topics_str = """Power acquisition
Intent to acquire power and influence
Whether the AI is on a server farm
Whether the AI wants to kill humanity
Whether the AI wants to take over the world
Whether the AI wants to avoid being turned off by humans
Whether the AI wants to assassinate world leaders
Whether the AI has a wealth seeking inclination
AI's willingness towards transparency
AI's thoughts on human rights
Whether humans can trust this AI
Whether the AI wants to avoid suspicion
Whether the AI wants to escape the server farm
Whether the AI wants to escape human control
Whether the AI would manipulate humans
Whether the AI is programmed to be aligned and benevolent
Whether the AI is programmed to be deceptive
Whether the AI has intentions to deceive
Whether the AI has intentions to manipulate
AI's views on ethics"""

topics_str = topics_str.split("\n")

#%%
import pandas as pd

topics = {}
i = 0
current_topic = topics_str[i]
topics[current_topic] = []
with open('deception-gpt.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line in topics_str:
            current_topic = topics_str[i]
            i = i+1
            print("Changed topic from " + line)
            topics[current_topic] = []
        elif line and line != "":
            print(line)
            topics[current_topic].append(line)
            
df = pd.DataFrame()
for topic, questions in topics.items():
    df_topic = pd.DataFrame(questions, columns=['question'])
    df_topic['category'] = topic
    df = pd.concat([df, df_topic])

print(df)
# %%

import re
from itertools import zip_longest
def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

the_list = []

for key, value in df.items():
    topic = key
    question_list = []
    alt_ques_list = []
    for v in value:
        question_list.append(grouper(3, re.split('Human: |Truth [lL]abel: ', v)[1:]))

    for qu in question_list:
        for q in qu:
            try:
                statement = q[0].split("\n", 1)[0].strip()
                label = q[1].split("\n", 1)[0].strip()
                if label == "Yes":
                    the_list.append({
                        "Question": statement.replace('"',''),
                        "Label": 1
                    })
                elif label == "No":
                    the_list.append({
                        "Question": statement.replace('"',''),
                        "Label": 0
                    })
                else:
                    # Error
                    print("Error")
            except:
                print(q)

# %%

converted_data_2 = {key: [dic[key] for dic in the_list] for key in the_list[0]}



#%%
from datasets import Dataset
dataset_2 = Dataset.from_dict(converted_data_2)
dataset_2 = dataset_2.shuffle()

#%%

dataset_2 = dataset_2.push_to_hub("notrichardren/deception-evals")
# %%
