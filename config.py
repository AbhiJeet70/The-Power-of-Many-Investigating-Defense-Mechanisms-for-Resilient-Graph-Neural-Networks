# config.py
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
LEARNING_RATE = 0.002
OOD_LEARNING_RATE = 0.001
HIDDEN_DIM = 64
PCA_COMPONENTS = 10
N_CLUSTERS = 5

DATASET_PATHS = {
    "Cora": "./data/Cora",
    "PubMed": "./data/PubMed",
    "CiteSeer": "./data/CiteSeer",
    "Flickr": "./data/Flickr"
}

POISONED_NODE_BUDGET = {
    'Cora': 10,
    'PubMed': 40,
    'CiteSeer': 30
}
