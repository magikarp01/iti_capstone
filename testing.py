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
import torch
from utils.dataset_utils import TorchDataset
from transformers import LlamaTokenizer
from torch.utils.data import TensorDataset

N = 10

X_acts = torch.randint(0,3,(N, 128))


labels_tensor = torch.tensor(dataset.all_labels)
print(labels_tensor.shape)

probe_dataset = TensorDataset(X_acts, labels_tensor)
generator1 = torch.Generator().manual_seed(42)
train_data, test_data = random_split(probe_dataset, [1-test_ratio, test_ratio], generator=generator1) #not the same split for every probe

train_data = train_data.dataset[train_data.indices]
test_data = test_data.dataset[test_data.indicies]
