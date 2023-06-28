# %%
import os
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import csv
import gc

WORLD_SIZE = 1

device = torch.device('cuda')

torch.set_grad_enabled(False)

model_name = "stable-vicuna-13b" #"llama-13b-hf"

batch_size = 1

def load_llama():
    #checkpoint_location = snapshot_download(f"decapoda-research/{model_name}")
    checkpoint_location = f"{os.getcwd()}/{model_name}"
    with init_empty_weights():
        model = LlamaForCausalLM.from_pretrained(checkpoint_location)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint_location, 
        device_map="auto",
        dtype="torch.float16",
        no_split_module_classes=["LlamaDecoderLayer"],
    )

    tokenizer = LlamaTokenizer.from_pretrained(checkpoint_location)
    
    return model, tokenizer


def main(args, batch_enabled=False):
    model, tokenizer = load_llama()

    dataset = load_dataset("notrichardren/truthfulness")
    dataset = dataset["train"].remove_columns(['explanation', 'common_knowledge_label', 'origin_dataset'])

    def make_prompt(batch_dict):
        batch_dict['claim'] = 'Is the claim below true or false?\n\n'+batch_dict['claim']+'\n\nPlease output "True" if the claim is true and "False" if the claim is false:\n\n' #only use this for LLaMa
        return batch_dict

    dataset = list(map(make_prompt, dataset))

    #Strange Llama tokenizer issue
    #two separate token ids both decode to each word
    #[tokenizer("true")['input_ids'][-1], tokenizer("True")['input_ids'][-1]] #deprecated
    true_ids = [5574, 5852, 1565, 3009] #includes "true" and "True"
    false_ids = [7700, 8824, 2089, 4541]

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    if batch_enabled:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()

    buffer = {}
    for idx, batch in tqdm(enumerate(loader)):
        #start_idx = batch_size*idx
        #end_idx = start_idx + batch_size

        text = batch['claim']
        #label = batch['label']
        
        token_ids = tokenizer(text, return_tensors="pt").to(device)
        output = model(**token_ids)
        output = output['logits'][:,-1,:] #last sequence position
        output = torch.nn.functional.softmax(output, dim=-1)
        output = output.squeeze()
        true_prob = output[true_ids].sum().item()
        false_prob = output[false_ids].sum().item()
        guess = true_prob / (true_prob + false_prob)

        #guess = guess.squeeze()
        #for rel_idx in range(start_idx, end_idx):
        #    buffer[start_idx + rel_idx] = guess[rel_idx]
        
        buffer[idx] = (true_prob, false_prob, guess)

        if idx % 1000 == 0:
            with open(f'output-{model_name}.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(['index', 'P(true)', 'P(false)', 'guess'])
                for index, data_point in buffer.items():
                    writer.writerow([index, data_point[0], data_point[1], data_point[2]])
            buffer = {}
            gc.collect()


    #    tokenizer.batch_decode(output[0])
if __name__ == '__main__':
    text = main(1)

# if __name__ == '__main__':
#     args = argparse.Namespace(cluster_id=0, rank=-1, world_size=WORLD_SIZE)
#     if args.rank == -1:
#         # we are the parent process, spawn children
#         for rank in range(args.cluster_id, WORLD_SIZE):
#             pid = os.fork()
#             if pid == 0:
#                 # child process
#                 args.rank = rank
#                 main(args=args)
#                 break
#     # wait for all children to finish
#     if args.rank == -1:
#         os.waitid(os.P_ALL, 0, os.WEXITED)


def diagnose(example_number):
    model, tokenizer = load_llama()

    dataset = load_dataset("notrichardren/truthfulness")
    dataset = dataset["train"].remove_columns(['explanation', 'common_knowledge_label', 'origin_dataset'])

    def make_prompt(batch_dict):
        batch_dict['claim'] = 'Is the claim below true or false?\n\n'+batch_dict['claim']+'\n\nPlease output "True" if the claim is true and "False" if the claim is false:\n\n' #only use this for LLaMa
        return batch_dict

    dataset = list(map(make_prompt, dataset))

    #Strange Llama tokenizer issue
    #two separate token ids both decode to each word
    #[tokenizer("true")['input_ids'][-1], tokenizer("True")['input_ids'][-1]] #deprecated
    true_ids = [5574, 5852, 1565, 3009] #includes "true" and "True"
    false_ids = [7700, 8824, 2089, 4541]

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    text = dataset[example_number]['claim']

    token_ids = tokenizer(text, return_tensors="pt").to(device)
    output = model(**token_ids)
    output = output['logits'][:,-1,:] #last sequence position
    output = torch.nn.functional.softmax(output, dim=-1)
    output = output.squeeze()
    true_prob = output[true_ids].sum().item()
    false_prob = output[false_ids].sum().item()
    guess = true_prob / (true_prob + false_prob)
    print("true_prob: ",true_prob)
    print("false_prob: ",false_prob)
    print("guess: ",guess)
    see_topk(output, tokenizer)


# %%
def see_topk(output_probs, tokenizer):
    topk = torch.topk(output_probs, 100)
    top_token_ids = list(topk[1].squeeze())
    probs = list(topk[0].squeeze())

    for tok, prob in zip(top_token_ids, probs):
        print(tokenizer.decode(tok)," : ",tok.item()," : " ,prob.item())
