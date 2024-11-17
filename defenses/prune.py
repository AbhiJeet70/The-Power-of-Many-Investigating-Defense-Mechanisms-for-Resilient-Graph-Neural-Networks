# prune.py
import torch
import torch.nn.functional as F

def defense_prune_edges(data, quantile_threshold=0.9):
    """
    Prunes edges based on adaptive cosine similarity between node features.

    Parameters:
    - data: PyG data object representing the graph.
    - quantile_threshold: Quantile to determine pruning threshold (e.g., 0.9 means pruning edges in the top 10% dissimilar).

    Returns:
    - data: Updated PyG data object with pruned edges.
    """
    features = data.x
    norm_features = F.normalize(features, p=2, dim=1)  # Normalize features
    edge_index = data.edge_index

    # Calculate cosine similarity for each edge
    src, dst = edge_index[0], edge_index[1]
    cosine_similarities = torch.sum(norm_features[src] * norm_features[dst], dim=1)

    # Adaptive threshold based on quantile of similarity distribution
    similarity_threshold = torch.quantile(cosine_similarities, quantile_threshold).item()

    # Keep edges with cosine similarity above the threshold
    pruned_mask = cosine_similarities >= similarity_threshold
    pruned_edges = edge_index[:, pruned_mask]

    # Update edge index with pruned edges
    data.edge_index = pruned_edges

    return data
