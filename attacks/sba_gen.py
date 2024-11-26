import torch
import torch.nn.functional as F
from attacks.trigger_generator import TriggerGenerator

def sba_gen_attack(data, poisoned_nodes, trigger_gen, trigger_size=5):
    """
    Subgraph-Based Attack - Gaussian with Trigger Generator.
    """
    data_poisoned = data.clone()

    # Generate average features and use Trigger Generator
    connected_nodes = [data.edge_index[0][data.edge_index[1] == node] for node in poisoned_nodes[:trigger_size]]
    avg_features = torch.stack([
        data.x[nodes].mean(dim=0) if len(nodes) > 0 else data.x.mean(dim=0) for nodes in connected_nodes
    ])
    trigger_features = trigger_gen(avg_features)

    # Update poisoned node features
    data_poisoned.x[poisoned_nodes[:trigger_size]] = trigger_features

    return data_poisoned
