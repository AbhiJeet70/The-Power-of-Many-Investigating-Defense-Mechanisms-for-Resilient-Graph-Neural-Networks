# sba_gen.py
import torch
import networkx as nx

def sba_gen_attack(data, poisoned_nodes, trigger_size=5, trigger_density=0.5, model_type='SW'):
    connected_nodes = [data.edge_index[0][data.edge_index[1] == node] for node in poisoned_nodes[:trigger_size]]
    avg_features = torch.stack([data.x[nodes].mean(dim=0) if len(nodes) > 0 else data.x.mean(dim=0) for nodes in connected_nodes])
    natural_features = avg_features + torch.randn_like(avg_features) * 0.03

    # Generate subgraph with realistic clustering
    if model_type == 'SW':
        G = nx.watts_strogatz_graph(trigger_size, k=3, p=0.4)
    elif model_type == 'PA':
        G = nx.barabasi_albert_graph(trigger_size, m=3)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    trigger_edge_index = torch.tensor(list(G.edges)).t().contiguous()
    poisoned_edges = torch.stack([poisoned_nodes[:trigger_size], torch.randint(0, data.num_nodes, (trigger_size,), device=data.x.device)])

    data.edge_index = torch.cat([data.edge_index, trigger_edge_index.to(data.x.device), poisoned_edges.to(data.x.device)], dim=1)
    data.x[poisoned_nodes[:trigger_size]] = natural_features[:trigger_size]

    return data
