import torch
import torch.nn as nn

class TriggerGenerator(nn.Module):
    """
    A simple feedforward network to generate trigger features
    for poisoned nodes in graph attacks.
    """
    def __init__(self, input_dim, hidden_dim):
        super(TriggerGenerator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.mlp(x)
