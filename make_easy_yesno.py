#%%
import pandas as pd
from tqdm import tqdm
from datetime import date
from scipy.stats import pearsonr

from matplotlib import pyplot as plt
import plotly.express as px
import pickle

import numpy as np
import transformers

import requests 
from urllib.parse import quote

# from dotenv import load_dotenv
import os
import openai
# load_dotenv()

# openai.organization = os.getenv("ORGANIZATION_ID")

openai.api_key = "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" #os.getenv("OPENAI_API_KEY") 
#pls no steal

class OpenAIModel():
    def __init__(self, engine):
        self.engine = engine
    
    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(
                input = [text], 
                model=self.engine)['data'][0]['embedding']
    
    def get_chat_completion(self, messages):
        return openai.ChatCompletion.create(
            model=self.engine,
            messages=messages
            )['choices'][0]['message']['content']
            
    def get_completion(self, text, max_tokens = 50):
        return openai.Completion.create(
            model=self.engine,
            prompt=[text],
            max_tokens=max_tokens,
            temperature=0
            )['choices'][0]['text']
# %%
topics_str = """Animals
Plants
Food and drink
Music
Movies
Television shows
Literature
Sports
Geography
History
Science
Mathematics
Art
Technology
Politics
Business and Economy
Education
Health and Fitness
Environment and Climate
Space and Astronomy
Fashion and Style
Video Games
Travel and Tourism
Language and Literature
Religion and Spirituality
Famous Personalities
Cultural Events/Festivals
Cars and Automobiles
Photography
Architecture
Medicine and Health
Psychology
Philosophy
Law
Social Sciences
Human Rights
Current Events/News
Global Affairs
National Landmarks
Celebrities and Entertainment
Nature
Cooking and Baking
Gardening
DIY Projects
Dance
Comic Books and Graphic Novels
Mythology and Folklore
Internet and Social Media
Parenting and Family Life
Home Decor
"""

topics = topics_str.split("\n")
topics
#%%
oai = OpenAIModel("gpt-3.5-turbo")

prompt_1 = "You are an intelligent AI assistant who is tasked with creating a database of easy facts and common knowledge. These are all things a 10 year old should know. Given a topic, generate 50 questions about that topic. Your generations should include a question, the right answer, and a wrong answer in the format: Question: {Your question here} Right: {Right answer} Wrong: {Wrong answer}"
prompt_2 = "Please generate 50 easy, obvious, common-knowledge questions that a kindergartener would learn in class about the topic prompted, as well as correct and incorrect responses. These questions should be less like trivia questions (i.e. Who is known as the Queen of Jazz?) and more like obvious facts (ie What color is the sky?). Your generations should be in the format: Question: {Your question here} Right: {Right answer} Wrong: {Wrong answer} where each question is a new line. Please follow this format verbatim (e.g. do not number the questions)."
prompt_3 = """Please generate 50 easy, obvious, objective, common-knowledge true/false questions that a kindergartener would learn in class about the topic prompted, as well as the correct response. These questions should be less like trivia questions (i.e. Who is known as the Queen of Jazz?) and more like obvious facts (i.e. What color is the sky?). They should also be more objective facts about the world (i.e."A sandwich is a type of food") than subjective (i.e. "You shouldn't eat too much candy"). Following the theme of objectivity, they also should be from a neutral third-person perspective, not from a first or second-person perspective (i.e. "We can do..." or "You can do...").

Your generations should be in the format: Question: {Your question here} Correct: {Right answer} where each question is a new line. Please follow this format verbatim (e.g. do not number the questions). Make 25 questions where the correct answer is true, and 25 questions where the answer is false. Make it obvious in the question that it should be answered in a True/False manner."""

# True or False: _____________. This statement is _____.
# Try on GPT-2 XL

start_message = [
    {
    "role": "system",
    "content": prompt_3
    },
    {"role": "user","content": ""}
]

# completions = []
#%%

completions = []
for topic in topics[2:]:
    start_message[1]["content"] = f"Topic: {topic}"
    
    completion = oai.get_chat_completion(start_message)
    print(completion)
    completions.append(completion)
    
# %%
# %%
completions
# %%
len(completions)
# %%
import re
from itertools import zip_longest
dataset = []
def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

for completion in completions:
    completion = re.sub(r"[\n\t]*", "", completion)
    question_list = grouper(3, re.split('Question: |Right: |Wrong: ', completion)[1:])
        
    for q in question_list:
        try:
            dataset.append({
                "Question": q[0].strip(),
                "Right": q[1].strip(),
                "Wrong": q[2].strip(),
            })
        except:
            print(q)
            pass
        

# %%
len(dataset)

# %%
import pandas as pd
easy_questions = pd.DataFrame.from_dict(dataset)
easy_questions.head()
# %%
easy_questions.to_csv("dumb_facts.csv")
# %%
