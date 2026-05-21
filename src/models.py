import torch
import torch.nn as nn


class FPNet(nn.Module):
    """
    PINN for solving the Fokker-Planck equation.
    """

    def __init__(self, input_dim=1, hidden_layers=4, neurons=20, output_transform='abs'):
        super().__init__()
        self.output_transform = output_transform

        layers = [nn.Linear(input_dim, neurons), nn.Tanh()]

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(neurons, 1))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """"
        Forward pass with selectable output transformation to ensure p(x) >= 0.
        """
        raw_output = self.network(x)

        if self.output_transform == 'abs':
            # Original baseline
            return torch.abs(raw_output)
        elif self.output_transform == 'softplus':
            # Smooth alternative
            return torch.nn.functional.softplus(raw_output )
        elif self.output_transform == 'exp':
            # Exponential alternative
            return torch.exp(raw_output)
        else:
            return raw_output