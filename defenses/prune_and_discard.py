# prune_and_discard.py
import torch
import torch.nn.functional as F

def defense_prune_and_discard_labels(data, quantile_threshold=0.2):
    """
    Prunes edges based on adaptive cosine similarity and discards labels of nodes connected by pruned edges selectively.

    Parameters:
    - data: PyG data object representing the graph.
    - quantile_threshold: Quantile threshold for cosine similarity pruning (e.g., 0.2 means pruning edges in the bottom 20%).

    Returns:
    - data: Updated PyG data object with pruned edges and selectively discarded labels.
    """
    features = data.x
    norm_features = F.normalize(features, p=2, dim=1)  # Normalize features using PyTorch
    edge_index = data.edge_index

    # Calculate cosine similarity for each edge
    src, dst = edge_index[0], edge_index[1]
    cosine_similarities = torch.sum(norm_features[src] * norm_features[dst], dim=1)

    # Use quantile to determine adaptive threshold for pruning
    adaptive_threshold = torch.quantile(cosine_similarities, quantile_threshold).item()

    # Mask edges with similarity below the adaptive threshold
    pruned_mask = cosine_similarities < adaptive_threshold
    pruned_edges = edge_index[:, ~pruned_mask]  # Retain edges that are above the threshold

    # Update edge index with pruned edges
    data.edge_index = pruned_edges

    # Selectively discard labels of nodes connected by many pruned edges
    pruned_src, pruned_dst = edge_index[:, pruned_mask]
    pruned_nodes_count = torch.bincount(torch.cat([pruned_src, pruned_dst]), minlength=data.num_nodes)

    # Only discard labels if the node has a high count of pruned edges
    threshold_count = int(torch.median(pruned_nodes_count).item())  # Use median count as a threshold
    nodes_to_discard = torch.where(pruned_nodes_count > threshold_count)[0]

    data.y[nodes_to_discard] = -1  # Use -1 to represent discarded labels

    return data
