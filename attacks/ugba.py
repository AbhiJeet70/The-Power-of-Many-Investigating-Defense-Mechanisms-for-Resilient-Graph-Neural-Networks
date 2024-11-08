# ugba.py
import torch
from clustering.dominant_set_clustering import dominant_set_clustering

def ugba_attack(data, poisoned_nodes, cluster_threshold=0.8, trigger_density=0.5):
    # Apply clustering to select diverse nodes within the poisoned nodes for higher stealth
    _, data = dominant_set_clustering(data, threshold=cluster_threshold)

    connected_nodes = [data.edge_index[0][data.edge_index[1] == node] for node in poisoned_nodes]
    avg_features = torch.stack([data.x[nodes].mean(dim=0) if len(nodes) > 0 else data.x.mean(dim=0) for nodes in connected_nodes])

    refined_trigger_features = avg_features + torch.normal(mean=2.0, std=0.5, size=avg_features.shape).to(data.x.device)
    data.x[poisoned_nodes] = refined_trigger_features

    # Add edges to reinforce connections
    new_edges = []
    for i in range(len(poisoned_nodes)):
        node = poisoned_nodes[i]
        neighbor = connected_nodes[i][0] if len(connected_nodes[i]) > 0 else poisoned_nodes[(i + 1) % len(poisoned_nodes)]
        new_edges.append([node, neighbor])

    new_edges = torch.tensor(new_edges, dtype=torch.long).t().contiguous().to(data.edge_index.device)
    data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)

    return data
