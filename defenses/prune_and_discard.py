# prune_and_discard.py
import torch
import torch.nn.functional as F

def defense_prune_and_discard_labels(data, quantile_threshold=0.2):
    features = data.x
    norm_features = F.normalize(features, p=2, dim=1)
    edge_index = data.edge_index
    src, dst = edge_index[0], edge_index[1]
    cosine_similarities = torch.sum(norm_features[src] * norm_features[dst], dim=1)
    adaptive_threshold = torch.quantile(cosine_similarities, quantile_threshold).item()
    pruned_mask = cosine_similarities < adaptive_threshold
    data.edge_index = edge_index[:, ~pruned_mask]

    pruned_src, pruned_dst = edge_index[:, pruned_mask]
    pruned_nodes_count = torch.bincount(torch.cat([pruned_src, pruned_dst]), minlength=data.num_nodes)
    threshold_count = int(torch.median(pruned_nodes_count).item())
    nodes_to_discard = torch.where(pruned_nodes_count > threshold_count)[0]
    data.y[nodes_to_discard] = -1
    return data
