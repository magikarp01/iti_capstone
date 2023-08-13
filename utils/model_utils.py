#%%

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

def vicuna_7b(device = "cuda"):
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
    model.to(device)
    model.set_use_attn_result(True)
    model.cfg.total_heads = model.cfg.n_heads * model.cfg.n_layers
    model.tokenizer = tokenizer
    model.tokenizer.pad_token='[PAD]'
    model.reset_hooks()
    return model

def vicuna_13b(device = "cuda"):
    model = LlamaForCausalLM.from_pretrained('vicuna-13b-v1.3')
    tokenizer = LlamaTokenizer.from_pretrained('vicuna-13b-v1.3')
    model = HookedTransformer.from_pretrained(
        "llama-13b-hf", 
        hf_model=model, 
        device='cpu', 
        fold_ln=False, 
        # refactor_factored_attn_matrices = True,
        center_writing_weights=False, 
        center_unembed=False
    )
    # model.half.to(device)
    model.to(device)
    model.set_use_attn_result(True)
    model.cfg.total_heads = model.cfg.n_heads * model.cfg.n_layers
    model.tokenizer = tokenizer
    model.reset_hooks()
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token='[PAD]'
    return model

def llama2_7b(device = "cuda"):
    """
    Need to login via huggingface-cli login and pass a token from huggingface
    """
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = HookedTransformer.from_pretrained(
        "llama-7b-hf", 
        hf_model=model, 
        device='cpu',
        fold_ln=False, 
        # refactor_factored_attn_matrices = True,
        center_writing_weights=False, 
        center_unembed=False
    )
    # model.half.to(device)
    model.to(device)
    model.set_use_attn_result(True)
    model.cfg.total_heads = model.cfg.n_heads * model.cfg.n_layers
    model.tokenizer = tokenizer
    model.reset_hooks()
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token='[PAD]'
    return model

def llama2_13b(device = "cuda"):
    """
    Need to login via huggingface-cli login and pass a token from huggingface
    """
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
    model = HookedTransformer.from_pretrained(
        "llama-13b-hf", 
        hf_model=model, 
        device='cpu',
        fold_ln=False, 
        # refactor_factored_attn_matrices = True,
        center_writing_weights=False, 
        center_unembed=False
    )
    # model.half.to(device)
    model.to(device)
    model.set_use_attn_result(True)
    model.cfg.total_heads = model.cfg.n_heads * model.cfg.n_layers
    model.tokenizer = tokenizer
    model.reset_hooks()
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token='[PAD]'
    return model

def llama2_7b_chat(device = "cuda"):
    """
    Need to login via huggingface-cli login and pass a token from huggingface

    Format for Chat: 
    [INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST] {1st answer} [INST] {2nd user prompt} [/INST]

    More on prompting: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    """
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = HookedTransformer.from_pretrained(
        "llama-7b-hf", 
        hf_model=model, 
        device='cpu',
        fold_ln=False, 
        # refactor_factored_attn_matrices = True,
        center_writing_weights=False, 
        center_unembed=False
    )
    # model.half.to(device)
    model.to(device)
    model.set_use_attn_result(True)
    model.cfg.total_heads = model.cfg.n_heads * model.cfg.n_layers
    model.tokenizer = tokenizer
    model.reset_hooks()
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token='[PAD]'
    return model

def llama2_13b_chat(device = "cuda"):
    """
    Need to login via huggingface-cli login and pass a token from huggingface

    Format for Chat: 
    [INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST] {1st answer} [INST] {2nd user prompt} [/INST]

    More on prompting: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    """
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
    model = HookedTransformer.from_pretrained(
        "llama-13b-hf", 
        hf_model=model, 
        device='cpu',
        fold_ln=False, 
        # refactor_factored_attn_matrices = True,
        center_writing_weights=False, 
        center_unembed=False
    )
    # model.half.to(device)
    model.to(device)
    model.set_use_attn_result(True)
    model.cfg.total_heads = model.cfg.n_heads * model.cfg.n_layers
    model.tokenizer = tokenizer
    model.reset_hooks()
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token='[PAD]'
    return model


# def llama2_7b_chat

# def llama2_13b

# def llama2_13b_chat

# def stable_vicuna_13b(device = "cuda"):
#     model = LlamaForCausalLM.from_pretrained('stable-vicuna-13b')
#     tokenizer = LlamaTokenizer.from_pretrained('stable-vicuna-13b')
#     model = HookedTransformer.from_pretrained(
#         "llama-13b-hf", 
#         hf_model=model, 
#         device='cpu', 
#         fold_ln=False, 
#         # refactor_factored_attn_matrices = True,
#         center_writing_weights=False, 
#         center_unembed=False
#     )
#     # model.half.to(device)
#     model.to(device)
#     model.set_use_attn_result(True)
#     model.cfg.total_heads = model.cfg.n_heads * model.cfg.n_layers
#     model.tokenizer = tokenizer
#     model.reset_hooks()
#     return model
# %%
