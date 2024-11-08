# trigger_generator.py
import torch

class TriggerGenerator(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TriggerGenerator, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.mlp(x)
