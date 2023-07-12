# %%
import torch
from transformers import GPT2Model, AutoTokenizer

model = GPT2Model.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# %%

text = "I have the power"
input_ids = torch.tensor(tokenizer(text)['input_ids'])


# %%
model.eval()
with torch.no_grad():
    print(model(input_ids))
