# load_data.py
from torch_geometric.datasets import Planetoid, Flickr
from config import DATASET_PATHS, DEVICE
import torch
import numpy as np

def load_dataset(dataset_name):
    if dataset_name in DATASET_PATHS:
        if dataset_name in ["Cora", "PubMed", "CiteSeer"]:
            return Planetoid(root=DATASET_PATHS[dataset_name], name=dataset_name)
        elif dataset_name == "Flickr":
            return Flickr(root=DATASET_PATHS[dataset_name])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def split_dataset(data, test_size=0.2, val_size=0.1):
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    num_test, num_val = int(test_size * num_nodes), int(val_size * num_nodes)
    num_train = num_nodes - num_test - num_val

    train_mask, val_mask, test_mask = torch.zeros(num_nodes, dtype=torch.bool), torch.zeros(num_nodes, dtype=torch.bool), torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:num_train]], val_mask[indices[num_train:num_train + num_val]], test_mask[indices[num_train + num_val:]] = True, True, True

    data.train_mask, data.val_mask, data.test_mask = train_mask.to(DEVICE), val_mask.to(DEVICE), test_mask.to(DEVICE)
    return data
