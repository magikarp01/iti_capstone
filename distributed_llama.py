import os
import torch
import torch.distributed as dist
from transformers import LlamaModel, LlamaForCausalLM, LlamaTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate import infer_auto_device_map
from huggingface_hub import snapshot_download



llama_two = "meta-llama/Llama-2-13b-chat-hf"
api_key = "thing" 
master_ip = "thing"
TOTAL_RANKS = 2


def main(args):

    dist.init_process_group(backend='nccl', init_method=f'tcp://{master_ip}:12345', world_size=TOTAL_RANKS, rank=args.rank)


    checkpoint_location = snapshot_download(llama_two, use_auth_token=api_key, local_dir=os.getcwd(), ignore_patterns=["*.safetensors", "model.safetensors.index.json"])
    with init_empty_weights():
        model = LlamaForCausalLM.from_pretrained(checkpoint_location)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint_location,
        device_map="auto",
        dtype=torch.float16,
        no_split_module_classes=["LlamaDecoderLayer"],
    )
    tokenizer = LlamaTokenizer.from_pretrained(checkpoint_location)



