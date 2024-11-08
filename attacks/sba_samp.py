# sba_samp.py
import torch

def sba_samp_attack(data, poisoned_nodes, trigger_gen):
    connected_nodes = [data.edge_index[0][data.edge_index[1] == node] for node in poisoned_nodes]
    avg_features = torch.stack([data.x[nodes].mean(dim=0) if len(nodes) > 0 else data.x.mean(dim=0) for nodes in connected_nodes])
    natural_features = avg_features + torch.randn_like(avg_features) * 0.02
    data.x[poisoned_nodes] = natural_features
    return data
