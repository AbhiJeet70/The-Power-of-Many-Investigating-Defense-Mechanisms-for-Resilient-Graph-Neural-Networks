# metrics.py
import torch
from sklearn.metrics import accuracy_score

def compute_metrics(model, data, poisoned_nodes):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        _, pred = out.max(dim=1)
        asr = (pred[poisoned_nodes] == data.y[poisoned_nodes]).sum().item() / len(poisoned_nodes) * 100
        clean_acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu()) * 100
    return asr, clean_acc
