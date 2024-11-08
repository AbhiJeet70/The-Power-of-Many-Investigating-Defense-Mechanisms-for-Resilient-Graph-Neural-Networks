# visualize.py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_pca_for_attacks(attack_embeddings_dict):
    pca = PCA(n_components=2)
    plt.figure(figsize=(20, 10))
    for i, (attack, attack_data) in enumerate(attack_embeddings_dict.items(), 1):
        embeddings = attack_data['data'].cpu().numpy()
        poisoned_nodes = attack_data['poisoned_nodes'].cpu().numpy()
        pca_result = pca.fit_transform(embeddings)
        clean_mask = ~poisoned_nodes
        plt.subplot(2, 3, i)
        plt.scatter(pca_result[clean_mask, 0], pca_result[clean_mask, 1], s=10, alpha=0.5, label='Clean Nodes', c='b')
        plt.scatter(pca_result[~clean_mask, 0], pca_result[~clean_mask, 1], s=10, alpha=0.8, label='Poisoned Nodes', c='r')
        plt.title(f'PCA Visualization for {attack}')
        plt.legend()
    plt.show()
