# dominant_set_clustering.py
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

def dominant_set_clustering(data, threshold=0.7, use_pca=True, pca_components=10, n_clusters=5):
    node_features = data.x.detach().cpu().numpy()
    if use_pca and node_features.shape[1] > pca_components:
        node_features = PCA(n_components=pca_components).fit_transform(node_features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(node_features)
    cluster_labels = kmeans.labels_
    distances = np.linalg.norm(node_features - kmeans.cluster_centers_[cluster_labels], axis=1)
    distance_threshold = np.percentile(distances, 100 * threshold)
    outliers = np.where(distances > distance_threshold)[0]
    data.y[outliers] = -1
    data.x[outliers] = data.x.mean(dim=0).to(data.x.device)
    return data
