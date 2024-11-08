# dpgba.py
import torch

def dpgba_attack(data, poisoned_nodes, trigger_gen, alpha=0.7):
    connected_nodes = [data.edge_index[0][data.edge_index[1] == node] for node in poisoned_nodes]
    avg_features = torch.stack([data.x[nodes].mean(dim=0) if len(nodes) > 0 else data.x.mean(dim=0) for nodes in connected_nodes])

    trigger_features = trigger_gen(avg_features)
    node_alphas = torch.rand(len(poisoned_nodes)).to(data.x.device) * 0.3 + 0.5
    distribution_preserved_features = node_alphas.unsqueeze(1) * data.x[poisoned_nodes] + (1 - node_alphas.unsqueeze(1)) * trigger_features

    data.x[poisoned_nodes] = distribution_preserved_features

    return data
