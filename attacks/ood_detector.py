import torch
from torch_geometric.nn import GCNConv

class OODDetector(torch.nn.Module):
    """
    OOD Detector for identifying poisoned nodes in DPGBA.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(OODDetector, self).__init__()
        self.encoder = torch.nn.Sequential(
            GCNConv(input_dim, hidden_dim),
            torch.nn.ReLU(),
            GCNConv(hidden_dim, latent_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, edge_index):
        z = self.encoder[0](x, edge_index)
        z = self.encoder[1](z)
        z = self.encoder[2](z, edge_index)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, z

    def reconstruction_loss(self, x, edge_index):
        reconstructed_x, _ = self.forward(x, edge_index)
        return torch.nn.functional.mse_loss(reconstructed_x, x, reduction='none').mean(dim=1)
