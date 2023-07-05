# %%
from utils.probing_utils import ModelActsLarge
from utils.model_loading_utils import load_llama
from utils.dataset_utils import TorchDataset


# %%
model_name = "stable-vicuna-13b"
model = load_llama(model_name)

dataset = TorchDataset("tqa", model.tokenizer)

acts = ModelActsLarge(model, dataset, use_aws=False,)

# %%

