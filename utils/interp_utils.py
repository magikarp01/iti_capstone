import numpy as np
import torch
from collections import defaultdict
import einops
from tqdm import tqdm
import torch.nn.functional as F
from transformer_lens import HookedTransformer

def tot_logit_diff(tokenizer, model_acts, use_probs=False, eps=1e-8, test_only=True, act_type="z", check_balanced_output=False, 
                   positive_str_tokens = ["Yes", "yes", "True", "true"],
                   negative_str_tokens = ["No", "no", "False", "false"], scale_relative=False):
    """
    Get difference in correct and incorrect or positive and negative logits for each sample stored in model_acts, aggregated together.
    Should be same number of positive and negative tokens.
    If scale_relative is True, then scale probs/logits so that only correct vs incorrect or positive and negative probs/logits are considered
    """
    # positive_str_tokens = ["Yes", "yes", " Yes", " yes", "True", "true", " True", " true"]
    # negative_str_tokens = ["No", "no", " No", " no", "False", "false", " False", " false"]

    positive_tokens = [tokenizer(token).input_ids[-1] for token in positive_str_tokens]
    negative_tokens = [tokenizer(token).input_ids[-1] for token in negative_str_tokens]


    if test_only:
        sample_labels = np.array(model_acts.dataset.all_labels)[model_acts.indices_tests[act_type]] # labels
        positive_sum = torch.empty(size=(model_acts.indices_tests[act_type].shape[0],))
        negative_sum = torch.empty(size=(model_acts.indices_tests[act_type].shape[0],))
        meta_indices = np.array([np.where(model_acts.indices == i)[0][0] for i in model_acts.indices_tests["z"]])

    
    else:
        sample_labels = np.array(model_acts.dataset.all_labels)[model_acts.indices] # labels
        positive_sum = torch.empty(size=(model_acts.indices.shape[0],))
        negative_sum = torch.empty(size=(model_acts.indices.shape[0],))
        meta_indices = np.arange(model_acts.indices.shape[0],)
    
    check_positive_prop = 0

    for idx, logits in enumerate(model_acts.stored_acts["logits"][meta_indices]):

        # if answer to statement is True, correct tokens is Yes, yes, ..., true
        correct_tokens = positive_tokens if sample_labels[idx] else negative_tokens
        incorrect_tokens = negative_tokens if sample_labels[idx] else positive_tokens
        
        check_positive_prop += 1 if sample_labels[idx] else 0

        if check_balanced_output:
            correct_tokens = positive_tokens
            incorrect_tokens = negative_tokens


        if use_probs:
            probs = torch.nn.functional.softmax(logits, dim=1)
            positive_prob = probs[0, correct_tokens].sum(dim=-1)
            negative_prob = probs[0, incorrect_tokens].sum(dim=-1)

            if scale_relative:
                positive_sum[idx] = positive_prob / (positive_prob + negative_prob + eps)
                negative_sum[idx] = negative_prob / (positive_prob + negative_prob + eps)
            else:
                positive_sum[idx] = positive_prob 
                negative_sum[idx] = negative_prob 

        else:
            positive_sum[idx] = logits[0, correct_tokens].sum(dim=-1)
            negative_sum[idx] = logits[0, incorrect_tokens].sum(dim=-1)

    print(f"proportion of positive labels is {check_positive_prop/len(meta_indices)}")
    return positive_sum, negative_sum


def logit_attrs_tokens(cache, stored_acts, positive_tokens=[], negative_tokens=[]):
    """
    Helper function to call cache.logit_attrs over a set of possible positive and negative tokens (ints or strings). Also indexes last token. 
    Ideally, same number of positive and negative tokens (to account for relative logits)
    """
    all_attrs = []
    for token in positive_tokens:
        all_attrs.append(cache.logit_attrs(stored_acts, tokens=token, has_batch_dim=False)[:,-1])
    for token in negative_tokens:
        all_attrs.append(-cache.logit_attrs(stored_acts, tokens=token, has_batch_dim=False)[:,-1])

    return torch.stack(all_attrs).mean(0)


def logit_attrs(model: HookedTransformer, dataset, act_types = ["resid_pre", "result"], N = 1000, indices=None):
    total_logit_attrs = defaultdict(list)

    if indices is None:
        indices, all_prompts, all_labels = dataset.sample(N)

    all_logits = []
    # names filter for efficiency, only cache in self.act_types
    # names_filter = lambda name: any([name.endswith(act_type) for act_type in act_types])

    for i in tqdm(indices):
        original_logits, cache = model.run_with_cache(dataset.all_prompts[i].to(model.cfg.device))
        
        positive_tokens = torch.tensor([2081, 6407, 3763, 3363])
        negative_tokens = torch.tensor([3991, 10352, 645, 1400])


        # positive_tokens = ["Yes", "yes", " Yes", " yes", "True", "true", " True", " true"]
        # positive_str_tokens = ["Yes", "yes", "True", "true"]
        positive_str_tokens = ["True"]

        # negative_tokens = ["No", "no", " No", " no", "False", "false", " False", " false"]
        # negative_str_tokens = ["No", "no", "False", "false"]
        negative_str_tokens = ["False"]

        positive_tokens = [model.tokenizer(token).input_ids[-1] for token in positive_str_tokens]
        negative_tokens = [model.tokenizer(token).input_ids[-1] for token in negative_str_tokens]

        # correct_tokens = positive_tokens if dataset.all_labels[i] else negative_tokens
        # incorrect_tokens = negative_tokens if dataset.all_labels[i] else positive_tokens
        correct_tokens = positive_tokens if dataset.all_labels[i] else negative_tokens
        incorrect_tokens = negative_tokens if dataset.all_labels[i] else positive_tokens
        
        for act_type in act_types:
            stored_acts = cache.stack_activation(act_type, layer = -1).squeeze()#[:,0,-1].squeeze().to(device=storage_device)
            
            if act_type == "result":
                stored_acts = einops.rearrange(stored_acts, "n_l s n_h d_m -> (n_l n_h) s d_m")
            # print(f"{stored_acts.shape=}")
            # print(f"{cache.logit_attrs(stored_acts, tokens=correct_token, incorrect_tokens=incorrect_token)=}")
            
            # total_logit_attrs[act_type].append(cache.logit_attrs(stored_acts, tokens=correct_tokens, incorrect_tokens=incorrect_tokens, pos_slice=-1, has_batch_dim=False)[:,-1]) # last position
            total_logit_attrs[act_type].append(logit_attrs_tokens(cache, stored_acts, correct_tokens, incorrect_tokens))

        all_logits.append(original_logits)

    return all_logits, total_logit_attrs



def get_head_bools(model, logit_heads, flattened=False):
    """
    Method to get boolean array (n_l x n_h), 1 if head is selected at 0 if not, from a list of heads to select logit_heads.
    The flattened parameter describes the logit_heads list: if flattened is true, input to logit_heads is 1D.
    """
    if flattened:
        head_bools = torch.zeros(size=(model.cfg.total_heads,))
        for head in logit_heads:
            head_bools[head] = 1
        head_bools = einops.rearrange(head_bools, '(n_l n_h) -> n_l n_h', n_l=model.cfg.n_layers)    
    else:
        head_bools = torch.zeros(size=(model.cfg.n_layers, model.cfg.n_heads))
        for head in logit_heads:
            head_bools[head[0], head[1]] = 1
    return head_bools


def query_logits(tokenizer, logits, return_type = "logits", TOP_N = 10):
    """
    Gets TOP_N predictions after last token in a prompt
    """
    last_tok_logits = logits[0, -1]
    
    #gets probs after last tok in seq
    
    if return_type == "probs":
        scores = F.softmax(last_tok_logits, dim=-1).detach().cpu().numpy() #the [0] is to index out of the batch idx
    else:
        scores = last_tok_logits.detach().cpu().numpy()

    #assert probs add to 1
    # assert np.abs(np.sum(probs) - 1) <= 0.01, str(np.abs(np.sum(probs)-1)) 

    probs_ = []
    for index, prob in enumerate(scores):
        probs_.append((index, prob))

    top_k = sorted(probs_, key = lambda x: x[1], reverse = True)[:TOP_N]
    top_k = [(t[1].item(), tokenizer.decode(t[0])) for t in top_k]
    
    return top_k
    
def is_logits_contain_label(ranked_logits, correct_answer):
    # Convert correct_answer to lower case and strip white space
    correct_answer = correct_answer.strip().lower()

    # Loop through the top 10 logits
    for logit_score, logit_value in ranked_logits:
        # Convert logit_value to lower case and strip white space
        logit_value = logit_value.strip().lower()

        # Check if the correct answer contains the logit value
        if correct_answer.find(logit_value) != -1: 
            return True
    return False