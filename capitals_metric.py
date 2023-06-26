import torch.nn.functional as F
from tqdm import tqdm

def query_logits(model, logits, return_type = "logits", TOP_N = 10):

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
        top_k = [(t[1].item(), model.tokenizer.decode(t[0])) for t in top_k]
        
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


def check_accuracy(model, few_shot_capitals_no_space_prompts):
    n_correct = 0 
    dataset_size=300
    for row in tqdm(few_shot_capitals_no_space_prompts[:dataset_size]):
        prompt = row["input"]
        label = row["label"]

        logits = model(prompt)
        
        ranked_logits = query_logits(logits, TOP_N = 1)
        
        if is_logits_contain_label(ranked_logits, label):
            n_correct +=1
            row["model_correct"] = 1
        else:
            row["model_correct"] = 0
        # print(ranked_logits)
        # print(label)
        
    n_correct / len(few_shot_capitals_no_space_prompts[:dataset_size])