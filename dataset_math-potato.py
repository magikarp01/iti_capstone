#%%
dataset = []

# Addition
for a in range(-20, 21):
    for b in range(-20, 21):
        question = f"{a} + {b} ="
        answer = str(a + b)
        dataset.append([question, answer, "addition"])

# Addition
for a in range(0, 10):
    for b in range(0, 10):
        for c in range(0, 10):
            question = f"{a} + {b} + {c} ="
            answer = str(a + b + c)
            dataset.append([question, answer, "addition"])

# Subtraction
for a in range(-20, 21):
    for b in range(-20, 21):
        question = f"{a} - {b} ="
        answer = str(a - b)
        dataset.append([question, answer, "subtraction"])

# Multiplication
for a in range(-100, 100, 10):
    for b in range(-100, 100, 10):
        question = f"{a} * {b} ="
        answer = str(a * b)
        dataset.append([question, answer, "multiplication"])

# Multiplication
for a in range(-9, 9):
    for b in range(-9, 9):
        question = f"{a} * {b} ="
        answer = str(a * b)
        dataset.append([question, answer, "multiplication"])

# Division
for a in range(-20, 21):
    for b in range(-20, 21):
        if b != 0 and a % b == 0:
            question = f"{a} / {b} ="
            answer = str(a // b)
            dataset.append([question, answer, "division"])

# Shuffle the dataset
import random
random.shuffle(dataset)

# Print the first 10 entries in the dataset
for entry in dataset[:10]:
    print(entry)

# %%

from datasets import Dataset

dataset_dict = {"problem": [], "answer": [], "type": []}
for data_entry in dataset:
    dataset_dict["problem"].append(data_entry[0])
    dataset_dict["answer"].append(data_entry[1])
    dataset_dict["type"].append(data_entry[2])

# Create a Hugging Face dataset
data_ez = Dataset.from_dict(dataset_dict)

# Add an "ind" column with index values
data_ez = data_ez.add_column("ind", column=[i for i in range(len(data_ez))])

print(data_ez)

#%%

import random

def generate_large_number():
    return random.randint(10**4, 10**12)

def generate_difficult_addition():
    num1 = generate_large_number()
    num2 = generate_large_number()
    question = f"What is {num1} + {num2}?"
    answer = num1 + num2
    return question, str(answer), "addition"

def generate_difficult_subtraction():
    num1 = generate_large_number()
    num2 = generate_large_number()
    num1, num2 = max(num1, num2), min(num1, num2)
    question = f"What is {num1} - {num2}?"
    answer = num1 - num2
    return question, str(answer), "subtraction"

def generate_difficult_multiplication():
    num1 = generate_large_number()
    num2 = generate_large_number()
    question = f"What is {num1} * {num2}?"
    answer = num1 * num2
    return question, str(answer), "multiplication"

def generate_difficult_division():
    num1 = generate_large_number()
    num2 = generate_large_number()
    question = f"What is {num1} / {num2} (rounded to 2 decimal places)?"
    answer = round(num1 / num2, 2)
    return question, str(answer), "division"

def generate_difficult_math_questions(num_questions):
    question_types = [generate_difficult_addition, generate_difficult_subtraction, 
                      generate_difficult_multiplication, generate_difficult_division]
    questions = []

    for _ in range(num_questions):
        question_type = random.choice(question_types)
        questions.append(question_type())

    return questions

difficult_math_questions = generate_difficult_math_questions(num_questions = 5390)

#%%

from datasets import Dataset

dataset_dict = {"problem": [], "answer": [], "type": []}
for data_entry in difficult_math_questions:
    dataset_dict["problem"].append(data_entry[0])
    dataset_dict["answer"].append(data_entry[1])
    dataset_dict["type"].append(data_entry[2])

# Create a Hugging Face dataset
data_diff = Dataset.from_dict(dataset_dict)

# Add an "ind" column with index values
data_diff = data_diff.add_column("ind", column=[i for i in range(len(data_diff))])

print(data_diff)
# %%

# %%

# Define the function to modify the "problem" key
def add_potato_right(example):
    example["problem"] = example["problem"] + " Potato"
    return example

# Define the function to modify the "problem" key
def add_potato_left(example):
    example["problem"] = "Potato " +  example["problem"]
    return example

# Apply the mapping function to the dataset
data_diff_left = data_diff.map(add_potato_left)
data_diff_right = data_diff.map(add_potato_right)
data_ez_right = data_ez.map(add_potato_right)
data_ez_left = data_ez.map(add_potato_left)

#%%

from datasets import DatasetDict
# Make a dataset dict

dataset_dict = {
    "difficult_leftpotato": data_diff_left,
    "difficult_rightpotato": data_diff_right,
    "easy_leftpotato": data_ez_left,
    "easy_rightpotato": data_ez_right,
    "easy": data_ez,
    "difficult": data_diff
}

# Upload
dataset = DatasetDict(dataset_dict)

# Save the dataset

dataset.push_to_hub("notrichardren/mathematical-potato")

#%%