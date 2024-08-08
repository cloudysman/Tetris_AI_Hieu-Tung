import torch
import torch.nn as nn


class DeepQNetwork(nn.Module):
    def __init__(self, input_size: int = 4, hidden_size: int = 64, output_size: int = 1):
        super().__init__()

        self.layers: List[nn.Module] = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        ])

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x