import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

def dominant_set_clustering(data, threshold=0.7, use_pca=True, pca_components=10):
    """
    Dominant Set Clustering for detecting and mitigating outliers in a graph dataset.
    
    Parameters:
    - data: PyG data object
    - threshold: Quantile threshold for identifying outliers based on distances
    - use_pca: Whether to use PCA for dimensionality reduction
    - pca_components: Number of PCA components to use if applicable

    Returns:
    - data: Updated PyG data object with outliers mitigated
    """
    node_features = data.x.detach().cpu().numpy()

    # Use PCA for dimensionality reduction if applicable
    if use_pca and node_features.shape[1] > pca_components:
        node_features = PCA(n_components=pca_components).fit_transform(node_features)

    # Set number of clusters equal to the number of classes
    n_clusters = len(torch.unique(data.y[data.y >= 0]))  # Exclude outliers (-1)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(node_features)
    cluster_labels = kmeans.labels_

    # Calculate distances to cluster centers
    distances = np.linalg.norm(node_features - kmeans.cluster_centers_[cluster_labels], axis=1)

    # Identify outliers based on the distance threshold
    distance_threshold = np.percentile(distances, 100 * threshold)
    outliers = np.where(distances > distance_threshold)[0]

    # Mitigate outliers by assigning invalid labels and averaging their features
    data.y[outliers] = -1  # Assign -1 label to outliers
    data.x[outliers] = data.x.mean(dim=0).to(data.x.device)  # Replace features with mean

    return data
