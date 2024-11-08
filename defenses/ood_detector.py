# ood_detector.py
import torch
import torch.nn.functional as F

class OODDetector(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(OODDetector, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 16),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(16, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_loss(self, x):
        return F.mse_loss(self.forward(x), x, reduction='none').mean(dim=1)
