# iti_capstone
Fork of wanff/iti_hack, developing for ARENA capstone

An implementation of Inference-Time Intervention (https://arxiv.org/pdf/2306.03341.pdf) in TransformerLens. This repository has utilities for specifying an arbitrary dataset for probing/ITI, generating activations, training probes, inserting ITI hooks into a HookedTransformer model, and evaluating the output before/after ITI using API calls to GPT-Judge models from OpenAI.

The goals of this project are to analyze truth from a mechanistic interpretability POV, and to determine the ability of both probing for truth and ITI to generalize beyond a particular dataset.

# Datasets
Users can use one of several of our own datasets (TruthfulQA, CounterFact, our own EZ-Dataset, Capitals, BoolQ) or define their own dataset, all found in the utils/dataset_utils.py file. Dataset classes have all_prompts and all_labels fields and a sample() method to sample statements that are either true or false.

Ideally, provided datasets are balanced with true and false statements.

# Activation Caching and Probing
Both activation caching and probing is done in the ModelActs class, defined in the utils/probing_utils.py file. ModelActs is a class to handle all preparation for probing and ITI. 

To use ModelActs, first initialize a ModelActs object with a model and a dataset from utils.dataset_utils. You can optionally include an act_types parameter of a list of activation types to cache (you can cache any activations from the provided HookedTransformer's hook_dict). (Example in iti_validations.py)

Then, generate activations (also stored in the ModelActs class) using the gen_acts method, which handles sampling from prompts/labels and caching. You can optionally store and load activations in a file. Activations are stored in the self.stored_acts dictionary (keys are act_types).

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