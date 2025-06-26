import torch
import gc
import pandas as pd
from torch.optim import Adam
from config import DEVICE as device, POISONED_NODE_BUDGET
from data.load_data import load_dataset
from data.split_data import split_dataset
from gnn_models.gcn import GCN
from gnn_models.graph_sage import GraphSAGE
from gnn_models.gat import GAT
from attacks.trigger_generator import TriggerGenerator
from defenses.prune import defense_prune_edges
from defenses.prune_and_discard import defense_prune_and_discard_labels
from defenses.dominant_set_clustering import dominant_set_clustering
from utils.seed import set_seed
from utils.metrics import compute_metrics
from utils.visualize import visualize_pca_for_attacks
from attacks.sba_samp import sba_samp_attack
from attacks.sba_gen import sba_gen_attack
from attacks.gta import gta_attack
from attacks.ugba import ugba_attack
from attacks.dpgba import dpgba_attack

# Set reproducibility
set_seed()

def run_all_attacks():
    datasets = ["Cora", "PubMed", "CiteSeer"]
    results_summary = []
    attack_embeddings_dict = {}

    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)
        data = split_dataset(dataset[0].to(device))
        input_dim = data.num_features
        output_dim = dataset.num_classes if isinstance(dataset.num_classes, int) else dataset.num_classes[0]

        # Dataset-specific poisoning budgets
        poisoned_node_budget = POISONED_NODE_BUDGET.get(dataset_name, 3)

        # Define GNN models
        model_types = ['GCN', 'GraphSage', 'GAT']

        for model_type in model_types:
            # ================================
            # Baseline training (no attack)
            # ================================
            if model_type == 'GCN':
                baseline_model = GCN(input_dim=input_dim, hidden_dim=256, output_dim=output_dim).to(device)
            elif model_type == 'GraphSage':
                baseline_model = GraphSAGE(input_dim=input_dim, hidden_dim=256, output_dim=output_dim).to(device)
            elif model_type == 'GAT':
                baseline_model = GAT(input_dim=input_dim, hidden_dim=256, output_dim=output_dim).to(device)

            baseline_optimizer = Adam(baseline_model.parameters(), lr=0.002)
            baseline_model.train()
            for epoch in range(200):
                baseline_optimizer.zero_grad()
                out = baseline_model(data.x, data.edge_index)
                loss = torch.nn.functional.cross_entropy(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                baseline_optimizer.step()

            baseline_model.eval()
            with torch.no_grad():
                out = baseline_model(data.x, data.edge_index)
                _, pred = out.max(dim=1)
                clean_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item() * 100

            results_summary.append({
                "Dataset": dataset_name,
                "Model": model_type,
                "Attack": "None",
                "Defense": "None",
                "ASR": "N/A",
                "Clean Accuracy": clean_acc
            })
            print(f"{dataset_name} | Model: {model_type} | Baseline (No Attack) | Clean Accuracy: {clean_acc:.2f}%")

            # ================================
            # Run attacks
            # ================================
            if model_type == 'GCN':
                model = GCN(input_dim=input_dim, hidden_dim=256, output_dim=output_dim).to(device)
            elif model_type == 'GraphSage':
                model = GraphSAGE(input_dim=input_dim, hidden_dim=256, output_dim=output_dim).to(device)
            elif model_type == 'GAT':
                model = GAT(input_dim=input_dim, hidden_dim=256, output_dim=output_dim).to(device)
            optimizer = Adam(model.parameters(), lr=0.002)
            trigger_gen = TriggerGenerator(input_dim=input_dim, hidden_dim=64).to(device)

            poisoned_nodes = torch.arange(poisoned_node_budget).to(device)

            attack_methods = {
                'SBA-Samp': sba_samp_attack,
                'SBA-Gen': sba_gen_attack,
                'GTA': gta_attack,
                'UGBA': ugba_attack,
                'DPGBA': dpgba_attack,
            }

            for attack_name, attack_fn in attack_methods.items():
                if attack_name == 'DPGBA':
                    data_poisoned = attack_fn(data, poisoned_nodes, trigger_gen, alpha=0.7)
                else:
                    data_poisoned = attack_fn(data, poisoned_nodes, trigger_gen)

                data_poisoned.x = data_poisoned.x.detach().clone()
                model.train()
                for epoch in range(200):
                    optimizer.zero_grad()
                    out = model(data_poisoned.x, data_poisoned.edge_index)
                    loss = torch.nn.functional.cross_entropy(
                        out[data_poisoned.train_mask], data_poisoned.y[data_poisoned.train_mask]
                    )
                    loss.backward()
                    optimizer.step()

                # Eval with no defense
                asr, clean_acc = compute_metrics(model, data_poisoned, poisoned_nodes)
                results_summary.append({
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Attack": attack_name,
                    "Defense": "None",
                    "ASR": asr,
                    "Clean Accuracy": clean_acc
                })
                print(f"{dataset_name} | Model: {model_type} | Attack: {attack_name} | Defense: None | ASR: {asr:.2f}%, Clean Acc: {clean_acc:.2f}%")

                # Defense 1: Dominant Set Outlier Detection (DSOD)
                data_poisoned_dsod = dominant_set_clustering(data_poisoned.clone(), threshold=0.9, use_pca=True, pca_components=10)
                asr_dsod, clean_acc_dsod = compute_metrics(model, data_poisoned_dsod, poisoned_nodes)
                results_summary.append({
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Attack": attack_name,
                    "Defense": "Dominant Set Outlier Detection",
                    "ASR": asr_dsod,
                    "Clean Accuracy": clean_acc_dsod
                })
                print(f"{dataset_name} | Model: {model_type} | Attack: {attack_name} | Defense: DSOD | ASR: {asr_dsod:.2f}%, Clean Acc: {clean_acc_dsod:.2f}%")

                # Defense 2: Prune
                data_poisoned_prune = defense_prune_edges(data_poisoned.clone(), quantile_threshold=0.8)
                asr_prune, clean_acc_prune = compute_metrics(model, data_poisoned_prune, poisoned_nodes)
                results_summary.append({
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Attack": attack_name,
                    "Defense": "Prune",
                    "ASR": asr_prune,
                    "Clean Accuracy": clean_acc_prune
                })
                print(f"{dataset_name} | Model: {model_type} | Attack: {attack_name} | Defense: Prune | ASR: {asr_prune:.2f}%, Clean Acc: {clean_acc_prune:.2f}%")

                # Defense 3: Prune + LD
                data_poisoned_prune_ld = defense_prune_and_discard_labels(data_poisoned.clone(), quantile_threshold=0.8)
                asr_prune_ld, clean_acc_prune_ld = compute_metrics(model, data_poisoned_prune_ld, poisoned_nodes)
                results_summary.append({
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Attack": attack_name,
                    "Defense": "Prune + LD",
                    "ASR": asr_prune_ld,
                    "Clean Accuracy": clean_acc_prune_ld
                })
                print(f"{dataset_name} | Model: {model_type} | Attack: {attack_name} | Defense: Prune + LD | ASR: {asr_prune_ld:.2f}%, Clean Acc: {clean_acc_prune_ld:.2f}%")

                # Store for PCA
                attack_embeddings_dict[f"{dataset_name}-{model_type}-{attack_name}"] = {
                    'data': data_poisoned_dsod.x,
                    'poisoned_nodes': poisoned_nodes
                }

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Save and summarize
    results_df = pd.DataFrame(results_summary)
    print("\nSummary of Attack Success Rate and Clean Accuracy Before and After Defenses:")
    print(results_df)
    results_df.to_csv("backdoor_attack_results_summary.csv", index=False)
    visualize_pca_for_attacks(attack_embeddings_dict)

if __name__ == "__main__":
    run_all_attacks()
