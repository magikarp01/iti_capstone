import os
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import LlamaForCausalLM, LlamaTokenizer




def load_llama(model_name, dtype="torch.float16"):
    #checkpoint_location = snapshot_download(f"decapoda-research/{model_name}")
    checkpoint_location = f"~/iti_capstone/{model_name}"
    with init_empty_weights():
        model = LlamaForCausalLM.from_pretrained(checkpoint_location)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint_location, 
        device_map="auto",
        dtype=dtype,
        no_split_module_classes=["LlamaDecoderLayer"],
    )

    tokenizer = LlamaTokenizer.from_pretrained(checkpoint_location)
    model.tokenizer = tokenizer
    return model