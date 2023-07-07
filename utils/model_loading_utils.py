import os
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import LlamaForCausalLM, LlamaTokenizer




def load_llama(model_name, model_dir, dtype="torch.float16"):
    #checkpoint_location = snapshot_download(f"decapoda-research/{model_name}")
    #MUST FIRST RUN SCRIPT load_stable_vicuna_13b.sh
    #hard code meta data in here (e.g. n_layers, d_head, etc)
    checkpoint_location = f"{model_dir}/{model_name}"
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