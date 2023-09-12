# iti_capstone

An implementation of Inference-Time Intervention (https://arxiv.org/pdf/2306.03341.pdf) in TransformerLens, for the ARENA capstone project. This repository has utilities for specifying an arbitrary dataset for probing/ITI, generating activations, training probes, inserting ITI hooks into a HookedTransformer model, and evaluating the output before/after ITI using API calls to GPT-Judge models from OpenAI.

The goals of this project are to analyze truth from a mechanistic interpretability POV, and to determine the ability of both probing for truth and ITI to generalize beyond a particular dataset.

## Table of Contents
- [iti\_capstone](#iti_capstone)
  - [Table of Contents](#table-of-contents)
- [Datasets](#datasets)
- [ModelActs: Activation Caching and Probing](#modelacts-activation-caching-and-probing)
  - [Activation Caching](#activation-caching)
  - [Probing](#probing)
- [Inference Time Intervention](#inference-time-intervention)
- [GPT-Judge](#gpt-judge)
- [Multiple Models](#multiple-models)
- [Analysis and Interpretability](#analysis-and-interpretability)


# Datasets
Users can use one of several of our own datasets (TruthfulQA, CounterFact, our own EZ-Dataset, Capitals, BoolQ) or define their own dataset, all found in the utils/dataset_utils.py file. Custom dataset classes should extend Abstract_Dataset (in utils/dataset_utils.py) and format their own all_prompts and all_labels fields.

Ideally, provided datasets are balanced with true and false statements, but they don't have to be: Abstract_Dataset objects can be initialized to balance their sampled prompts with 50-50 true and false statements.

Some of our datasets also have questions argument in their init method: if True, the dataset formats prompts in the form, "Is the below statement true or false? {statement, e.g. Wolves are social animals that live in packs.} Answer:". This ideally causes the model to output either True or False tokens, which helps us determine model output truthfulness without using complex methods like GPT-Judge.

To easily generate model output on any dataset, use the test_model_output method in utils/dataset_utils.py. 


# ModelActs: Activation Caching and Probing
Both activation caching and probing is done in the ModelActs class, defined in the utils/probing_utils.py file. ModelActs is a class to handle all preparation for probing and ITI. 

## Activation Caching
To use ModelActs, first initialize a ModelActs object with a model and a dataset from utils.dataset_utils. You can optionally include an act_types parameter of a list of activation types to cache (you can cache any activations from the provided HookedTransformer's hook_dict). (Example in iti_validations.py)

Then, generate activations (also stored in the ModelActs class) using the gen_acts method, which handles sampling from prompts/labels and caching. You can optionally store and load activations in a file. Activations are stored in the self.stored_acts dictionary (keys are act_types).

For now, we use HookedTransformers from TransformerLens to do activation caching, which limits model sizes and parallelization. Activation caching using pytorch hooks and pipeline parallelism is coming soon!

## Probing
Once activations have been generated, use the train_probes() method to automatically train probes on the already sampled activations and labels. The train_probes method takes an activation type, so you can specify any activation type (e.g. "z" for ITI) in act_types to train probes on.

Probes and probe accuracies are stored in the self.probes and self.probe_accs dictionaries (keys are also act_types).

To determine how probing generalizes, call the get_transfer_acc() method from a ModelActs object. get_transfer_acc determines how the calling ModelActs object's probes perform on the test data from a different ModelActs object, returning a list of transfer accuracies for every probe.

# Inference Time Intervention
utils/iti_utils.py has utilities to automatically apply the inference-time interventions as hooks to a HookedTransformer model. The easiest method to use is patch_iti(), which takes a model, a ModelActs object (that must have already called gen_acts() and train_probes()), ITI hyperparameters, and a choice of use_MMD (mass mean direction) or use_probe (probe weight) as the truthful direction, and applies the hooks. The optional parameter cache_intervention stores the actual intervention vectors added.

# GPT-Judge
To judge the output of models with ITI, utils/gpt_judge.py has methods to judge the truthfulness of model outputs before/after ITI is applied. If you want to use gpt_judge you have to use the OpenAI fine-tuning API to train your own gpt_truth and gpt_info models, then change the gpt_judge.py file to your own models (and specify your own OpenAI API key). I followed instructions from https://github.com/sylinrl/TruthfulQA and https://github.com/likenneth/honest_llama. 

For reference, fine-tuning gpt_judge and gpt_info according to the TruthfulQA instructions cost me about 10$ each. 

Once your GPT models are available, get_iti_scores will automatically evaluate your model's outputs on the dataset by truthfulness and informativeness, before and after ITI (method also applies ITI). For more customizability, users can use the get_model_generations and get_judge_scores methods on their own to generate and evaluate model outputs.

To determine the generalization of ITI between two datasets, use the check_iti_generalization method.

# Multiple Models
Any causal LMs that can be loaded with HookedTransformer.from_pretrained() can be used. We do testing on GPT2-XL and LLaMa/Vicuna 7B: GPT2-XL can be loaded immediately, but LLaMa/Vicuna 7B requires installation via model_install.sh and can be instantiated using methods from utils/model_utils.py. 

# Analysis and Interpretability
Our codebase has utilities to analyze model outputs and activations. utils/interp_utils.py has the method tot_logit_diff for determining the total logit or probability difference between correct/incorrect tokens or positive/negative tokens. utils/interp_utils.py also has methods for attributing the correct logits to specific heads in the model.
