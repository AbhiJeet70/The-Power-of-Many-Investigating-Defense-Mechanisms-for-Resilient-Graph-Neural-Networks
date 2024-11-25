import torch
from torch_geometric.datasets import Planetoid, Flickr

def load_dataset(dataset_name):
    if dataset_name in ["Cora", "PubMed", "CiteSeer"]:
        dataset = Planetoid(root=f"./data/{dataset_name}", name=dataset_name)
    elif dataset_name == "Flickr":
        dataset = Flickr(root="./data/Flickr")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return dataset
