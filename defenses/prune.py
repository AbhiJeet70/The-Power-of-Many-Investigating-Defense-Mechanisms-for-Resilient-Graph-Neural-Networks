# prune.py
import torch
import torch.nn.functional as F

def defense_prune_edges(data, quantile_threshold=0.9):
    features, norm_features = data.x, F.normalize(data.x, p=2, dim=1)
    src, dst = data.edge_index[0], data.edge_index[1]
    cosine_similarities = torch.sum(norm_features[src] * norm_features[dst], dim=1)
    similarity_threshold = torch.quantile(cosine_similarities, quantile_threshold).item()
    pruned_mask = cosine_similarities >= similarity_threshold
    data.edge_index = data.edge_index[:, pruned_mask]
    return data
