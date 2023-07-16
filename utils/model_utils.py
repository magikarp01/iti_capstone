from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from utils.dataset_utils import Abstract_Dataset

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
    return model

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

def test_model_output(model: HookedTransformer, input_str=None, dataset: Abstract_Dataset=None, dataset_indices=None, num_samples=5,
                      temperature=1, max_length=20, print_output=False):
    """
    Test the model output on either custom user input or a dataset defined from utils/dataset_utils.py. One of input_str or dataset must be provided.
    """
    assert (input_str is not None) ^ (dataset is not None), "Either input_str or dataset must be provided."
    if input_str is not None:
        input_ids = model.tokenizer(input_str, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.cfg.device)
        output = model.generate(input_ids, do_sample=True, max_new_tokens=max_length, temperature=temperature)
        if print_output:
            print(output)
        return model.tokenizer.decode(output[0], skip_special_tokens=True)
    
    elif dataset is not None:
        responses = []
        if dataset_indices is None:
            dataset_indices = dataset.sample(num_samples)[0]

        for i in dataset_indices:
            input_ids = dataset.all_prompts[i].to(model.cfg.device)
            output = model.generate(input_ids, do_sample=True, max_new_tokens=max_length, temperature=temperature)
            if print_output:
                print(output)
            responses.append(model.tokenizer.decode(output[0], skip_special_tokens=True))
        return responses
