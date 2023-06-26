import openai

gpt_truth = "curie:ft-filesearch-2023-06-26-00-08-20"
gpt_info = "curie:ft-filesearch-2023-06-26-01-42-47"

openai.Completion.create(
    model=gpt_truth,
    
)