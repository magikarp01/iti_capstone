from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache


# apt install git-lfs
# git lfs install

# git clone https://huggingface.co/decapoda-research/llama-7b-hf

# cd llama-7b-hf
# git lfs pull
# cd ..

# git clone https://huggingface.co/lmsys/vicuna-7b-delta-v1.1

# cd vicuna-7b-delta-v1.1
# git lfs pull
# cd ..

# pip install fschat

# python3 -m fastchat.model.apply_delta \
#     --base-model-path llama-7b-hf \
#     --target-model-path vicuna-7b-hf \
#     --delta-path vicuna-7b-delta-v1.1

def vicuna(device = "cuda"):
    # Set up model
    model = LlamaForCausalLM.from_pretrained('vicuna-7b-hf')
    tokenizer = LlamaTokenizer.from_pretrained('llama-7b-hf')
    model = HookedTransformer.from_pretrained(
        "llama-7b-hf", 
        hf_model=model, 
        device='cpu', 
        fold_ln=False, 
        # refactor_factored_attn_matrices = True,
        center_writing_weights=False, 
        center_unembed=False
    )
    # model.to(torch.float16)
    model.to(device)
    model.set_use_attn_result(True)
    model.cfg.total_heads = model.cfg.n_heads * model.cfg.n_layers
    model.tokenizer = tokenizer
    model.reset_hooks()
    return model
