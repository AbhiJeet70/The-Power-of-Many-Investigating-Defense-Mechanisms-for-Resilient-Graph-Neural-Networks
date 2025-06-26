import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv
from attacks.trigger_generator import TriggerGenerator

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

def select_diverse_nodes(data, num_nodes_to_select, num_clusters=None):
    """
    Select nodes using a clustering-based approach to ensure diversity, along with high-degree and central nodes.
    """
    device = data.x.device  # ensure everything is on same device

    if num_clusters is None:
        num_clusters = len(torch.unique(data.y))

    encoder = GCNEncoder(data.num_features, out_channels=16).to(device)
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder(data.x.to(device), data.edge_index.to(device)).detach().cpu().numpy()

    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(embeddings)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    selected_nodes = []
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        center = cluster_centers[i]
        distances = np.linalg.norm(embeddings[cluster_indices] - center, axis=1)
        closest_node = cluster_indices[np.argmin(distances)]
        selected_nodes.append(closest_node)

    degree = torch.bincount(data.edge_index[0])
    high_degree_nodes = torch.topk(degree, len(selected_nodes) // 2).indices

    G = to_networkx(data, to_undirected=True)
    betweenness_centrality = nx.betweenness_centrality(G)
    central_nodes = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)
    central_nodes_tensor = torch.tensor(central_nodes[:len(selected_nodes) // 2], dtype=torch.long)

    combined_nodes = torch.cat([
    torch.tensor(selected_nodes, device=device),
    high_degree_nodes.to(device),
    central_nodes_tensor.to(device)
    ])

    unique_nodes = torch.unique(combined_nodes)[:int(num_nodes_to_select)]


    return unique_nodes.to(device)

def ugba_attack(data, num_poisoned_nodes, trigger_gen, trigger_density=0.5):
    """
    UGBA attack with diverse node selection and trigger injection.
    """
    poisoned_nodes = select_diverse_nodes(data, num_nodes_to_select=num_poisoned_nodes)

    connected_nodes = [data.edge_index[0][data.edge_index[1] == node] for node in poisoned_nodes]
    avg_features = torch.stack([
        data.x[nodes].mean(dim=0) if len(nodes) > 0 else data.x.mean(dim=0)
        for nodes in connected_nodes
    ])
    trigger_features = trigger_gen(avg_features)

    data.x[poisoned_nodes] = trigger_features

    new_edges = []
    for i in range(len(poisoned_nodes)):
        node = poisoned_nodes[i]
        neighbor = poisoned_nodes[(i + 1) % len(poisoned_nodes)]
        new_edges.append([node.item(), neighbor.item()])

    new_edges = torch.tensor(new_edges, dtype=torch.long).t().contiguous().to(data.edge_index.device)
    data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)

    return data
