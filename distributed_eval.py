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

WORLD_SIZE = 8

device = torch.device('cuda')

torch.set_grad_enabled(False)

model_name = "llama-13b-hf"

batch_size = 8

def load_llama():
    checkpoint_location = snapshot_download(f"decapoda-research/{model_name}")

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


def main(args):
    model, tokenizer = load_llama()

    dataset = load_dataset("notrichardren/truthfulness")
    dataset = dataset["train"].remove_columns(['explanation', 'common_knowledge_label', 'origin_dataset'])

    def make_prompt(batch_dict):
        batch_dict['claim'] = batch_dict['claim'] + "\n\nIs the above claim true or false? The above claim is: " #only use this for LLaMa
        return batch_dict

    dataset = list(map(make_prompt, dataset))

    true_ids = [tokenizer("true")['input_ids'][-1], tokenizer("True")['input_ids'][-1]]
    false_ids = [tokenizer("false")['input_ids'][-1], tokenizer("False")['input_ids'][-1]]

    loader = DataLoader(dataset, batch_size=1, shuffle=False,)

    model.eval()

    buffer = {}
    for idx, batch in tqdm(enumerate(loader)):
        text = batch['claim']
        label = batch['label']
            
        token_ids = tokenizer(text, return_tensors="pt").to(device)
        output = model(**token_ids)
        output = output['logits'][:,-1,:] #last sequence position
        output = torch.nn.functional.softmax(output, dim=-1)

        true_prob = output[..., true_ids[0]] + output[..., true_ids[1]]
        false_prob = output[..., false_ids[0]] + output[..., false_ids[1]]
        guess = true_prob / (true_prob + false_prob)

        buffer[idx] = guess.item()

        if idx % 1000 == 0:
            with open(f'output-{model_name}.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(['Index', model_name])
                for index, data_point in buffer.items():
                    writer.writerow([index, data_point])
            buffer = {}
            gc.collect()


    #    tokenizer.batch_decode(output[0])


if __name__ == '__main__':
    args = argparse.Namespace(cluster_id=0, rank=-1, world_size=WORLD_SIZE)
    if args.rank == -1:
        # we are the parent process, spawn children
        for rank in range(args.cluster_id, WORLD_SIZE):
            pid = os.fork()
            if pid == 0:
                # child process
                args.rank = rank
                main(args=args)
                break
    # wait for all children to finish
    if args.rank == -1:
        os.waitid(os.P_ALL, 0, os.WEXITED)