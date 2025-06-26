import torch
from attacks.trigger_generator import TriggerGenerator
import torch.nn as nn
from torch_geometric.nn import GCNConv

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
    Select nodes using a clustering-based approach to ensure diversity, along with high-degree nodes.

    Parameters:
    - data: PyG data object representing the graph.
    - num_nodes_to_select: Number of nodes to select for poisoning.
    - num_clusters: Number of clusters to form for diversity. Defaults to number of classes if not provided.

    Returns:
    - Tensor containing indices of selected nodes.
    """
    # Set the number of clusters equal to the number of classes in the dataset if not provided
    if num_clusters is None:
        num_clusters = len(torch.unique(data.y))

    # Use GCN encoder to get node embeddings that capture both attribute and structural information
    encoder = GCNEncoder(data.num_features, out_channels=16)  # Assuming out_channels = 16
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder(data.x, data.edge_index).cpu().numpy()

    # Perform K-means clustering to find representative nodes
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(embeddings)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Select nodes closest to the cluster centers
    selected_nodes = []
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        center = cluster_centers[i]
        distances = np.linalg.norm(embeddings[cluster_indices] - center, axis=1)
        closest_node = cluster_indices[np.argmin(distances)]
        selected_nodes.append(closest_node)

    # Calculate node degrees
    degree = torch.bincount(data.edge_index[0])  # Calculate node degrees
    # Select high-degree nodes
    high_degree_nodes = torch.topk(degree, len(selected_nodes) // 2).indices

    # Convert the graph to NetworkX to calculate centrality measures
    G = to_networkx(data, to_undirected=True)
    betweenness_centrality = nx.betweenness_centrality(G)
    central_nodes = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)
    central_nodes_tensor = torch.tensor(central_nodes[:len(selected_nodes) // 2], dtype=torch.long)

    # Combine diverse nodes, high-degree nodes, and central nodes
    combined_nodes = torch.cat([torch.tensor(selected_nodes), high_degree_nodes, central_nodes_tensor])
    # Get unique nodes and limit to num_nodes_to_select
    unique_nodes = torch.unique(combined_nodes)[:num_nodes_to_select]

    return torch.tensor(unique_nodes, dtype=torch.long).to(data.x.device)

def ugba_attack(data, num_poisoned_nodes, trigger_gen, trigger_density=0.5):
    """
    UGBA attack that uses a simple trigger generator for poisoning.

    Parameters:
    - data: PyG data object
    - num_poisoned_nodes: Number of nodes to poison
    - trigger_gen: Trigger generator model
    - trigger_density: Density of the trigger subgraph

    Returns:
    - Modified PyG data object with poisoned nodes
    """
    # Select diverse nodes to poison
    poisoned_nodes = select_diverse_nodes(data, num_nodes_to_select=num_poisoned_nodes)

    # Generate trigger features using the trigger generator
    connected_nodes = [data.edge_index[0][data.edge_index[1] == node] for node in poisoned_nodes]
    avg_features = torch.stack([data.x[nodes].mean(dim=0) if len(nodes) > 0 else data.x.mean(dim=0) for nodes in connected_nodes])
    trigger_features = trigger_gen(avg_features)

    # Update poisoned nodes with generated trigger features
    data.x[poisoned_nodes] = trigger_features

    # Add edges to strengthen connections between poisoned nodes
    new_edges = []
    for i in range(len(poisoned_nodes)):
        node = poisoned_nodes[i]
        neighbor = poisoned_nodes[(i + 1) % len(poisoned_nodes)]  # Connect to the next poisoned node circularly
        new_edges.append([node, neighbor])

    # Convert new edges to tensor and add them to the graph
    new_edges = torch.tensor(new_edges, dtype=torch.long).t().contiguous().to(data.edge_index.device)
    data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)

    return data

