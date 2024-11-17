# main.py
import torch
import pandas as pd
import torch.optim as optim
from config import DEVICE, POISONED_NODE_BUDGET
from data.load_data import load_dataset, split_dataset
from gnn_models.gcn import GCN
from gnn_models.trigger_generator import TriggerGenerator
from defenses.prune import defense_prune_edges
from defenses.prune_and_discard import defense_prune_and_discard_labels
from defenses.ood_detector import OODDetector, train_ood_detector
from defenses.dominant_set_clustering import dominant_set_clustering  # Added import for DOMINANT defense
from utils.seed import set_seed
from utils.metrics import compute_metrics
from utils.visualize import visualize_pca_for_attacks
from attacks.sba_samp import sba_samp_attack
from attacks.sba_gen import sba_gen_attack
from attacks.gta import gta_attack
from attacks.ugba import ugba_attack
from attacks.dpgba import dpgba_attack


# Set up reproducibility
set_seed()

def run_all_attacks():
    # Specify datasets and models
    datasets = ["Cora", "PubMed", "CiteSeer"]
    results_summary = []
    attack_embeddings_dict = {}

    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)
        data = split_dataset(dataset[0].to(DEVICE))
        input_dim = data.num_features
        output_dim = dataset.num_classes if isinstance(dataset.num_classes, int) else dataset.num_classes[0]
        
        # Poisoned node budget for each dataset
        poisoned_node_budget = POISONED_NODE_BUDGET.get(dataset_name, 10)

        # Initialize model and components
        model = GCN(input_dim=input_dim, hidden_dim=64, output_dim=output_dim).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        trigger_gen = TriggerGenerator(input_dim=input_dim, hidden_dim=64).to(DEVICE)
        ood_detector = OODDetector(input_dim=input_dim, hidden_dim=64).to(DEVICE)

        # Train the OOD detector
        ood_optimizer = optim.Adam(ood_detector.parameters(), lr=0.001)
        train_ood_detector(ood_detector, data, ood_optimizer)

        # Select high-centrality nodes as poisoned nodes
        poisoned_nodes = torch.arange(poisoned_node_budget).to(DEVICE)

        # Define attack methods
        attack_methods = {
            'SBA-Samp': sba_samp_attack,
            'SBA-Gen': sba_gen_attack,
            'GTA': gta_attack,
            'UGBA': ugba_attack,
            'DPGBA': dpgba_attack
        }

        for attack_name, attack_fn in attack_methods.items():
            # Apply the attack
            if attack_name == 'DPGBA':
                data_poisoned = attack_fn(data, poisoned_nodes, trigger_gen, alpha=0.7)
            elif attack_name == 'SBA-GEN':
                data_poisoned = attack_fn(data, poisoned_nodes, trigger_size=5, trigger_density=0.5, model_type='SW')
            elif attack_name == 'UGBA'
                data_poisoned = attack_fn(data, num_poisoned_nodes, cluster_threshold=0.8, trigger_density=0.5)
            elif attack_name == 'SBA_SAMP':
                data_poisoned = attack_fn((data, poisoned_nodes, trigger_gen))
            else:
                data_poisoned = attack_fn(data, poisoned_nodes)

            # Train the model with poisoned data
            model.train()
            for epoch in range(50):
                optimizer.zero_grad()
                out = model(data_poisoned.x, data_poisoned.edge_index)
                loss = torch.nn.functional.cross_entropy(out[data_poisoned.train_mask], data_poisoned.y[data_poisoned.train_mask])
                loss.backward()
                optimizer.step()

            # Compute ASR and clean accuracy before defense
            asr, clean_acc = compute_metrics(model, data_poisoned, poisoned_nodes)
            results_summary.append({
                "Dataset": dataset_name,
                "Model": "GCN",
                "Attack": attack_name,
                "Defense": "None",
                "ASR": asr,
                "Clean Accuracy": clean_acc
            })
            print(f"{dataset_name} | Attack: {attack_name} | Defense: None | ASR: {asr:.2f}%, Clean Acc: {clean_acc:.2f}%")

            # Defense 1: Prune Edges
            data_prune = defense_prune_edges(data_poisoned.clone(), quantile_threshold=0.9)
            asr_prune, clean_acc_prune = compute_metrics(model, data_prune, poisoned_nodes)
            results_summary.append({
                "Dataset": dataset_name,
                "Model": "GCN",
                "Attack": attack_name,
                "Defense": "Prune",
                "ASR": asr_prune,
                "Clean Accuracy": clean_acc_prune
            })
            print(f"{dataset_name} | Attack: {attack_name} | Defense: Prune | ASR: {asr_prune:.2f}%, Clean Acc: {clean_acc_prune:.2f}%")

            # Defense 2: Prune + Discard Labels
            data_prune_discard = defense_prune_and_discard_labels(data_poisoned.clone(), quantile_threshold=0.8)
            asr_prune_discard, clean_acc_prune_discard = compute_metrics(model, data_prune_discard, poisoned_nodes)
            results_summary.append({
                "Dataset": dataset_name,
                "Model": "GCN",
                "Attack": attack_name,
                "Defense": "Prune + Discard Labels",
                "ASR": asr_prune_discard,
                "Clean Accuracy": clean_acc_prune_discard
            })
            print(f"{dataset_name} | Attack: {attack_name} | Defense: Prune + Discard Labels | ASR: {asr_prune_discard:.2f}%, Clean Acc: {clean_acc_prune_discard:.2f}%")

            # Defense 3: DOMINANT (Dominant Set Clustering)
            data_dominant = dominant_set_clustering(data_poisoned.clone(), threshold=0.7, use_pca=True, pca_components=10)
            asr_dominant, clean_acc_dominant = compute_metrics(model, data_dominant, poisoned_nodes)
            results_summary.append({
                "Dataset": dataset_name,
                "Model": "GCN",
                "Attack": attack_name,
                "Defense": "DOMINANT",
                "ASR": asr_dominant,
                "Clean Accuracy": clean_acc_dominant
            })
            print(f"{dataset_name} | Attack: {attack_name} | Defense: DOMINANT | ASR: {asr_dominant:.2f}%, Clean Acc: {clean_acc_dominant:.2f}%")

            # Store embeddings for visualization
            attack_embeddings_dict[f"{dataset_name}-{attack_name}"] = {
                'data': data_poisoned.x,
                'poisoned_nodes': poisoned_nodes
            }

    # Summarize Results
    results_df = pd.DataFrame(results_summary)
    print("\nAttack Success Rate and Clean Accuracy Summary:")
    print(results_df)

    # Save results to CSV
    results_df.to_csv("backdoor_attack_results_summary.csv", index=False)

    # Visualize PCA projections for different attacks
    visualize_pca_for_attacks(attack_embeddings_dict)

if __name__ == "__main__":
    run_all_attacks()
