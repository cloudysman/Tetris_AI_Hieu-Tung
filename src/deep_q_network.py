import torch
import torch.nn as nn
from typing import List, Union

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
    
    @classmethod
    def from_old_model(cls, old_model: Union[dict, 'DeepQNetwork']):
        new_model = cls()
        
        if isinstance(old_model, dict):
            old_state_dict = old_model
        elif hasattr(old_model, 'state_dict'):
            old_state_dict = old_model.state_dict()
        else:
            raise ValueError("Unsupported model type")

        new_state_dict = new_model.state_dict()

        # Map old keys to new keys
        old_to_new = {
            'conv1.0.weight': 'layers.0.weight',
            'conv1.0.bias': 'layers.0.bias',
            'conv2.0.weight': 'layers.2.weight',
            'conv2.0.bias': 'layers.2.bias',
            'conv3.0.weight': 'layers.4.weight',
            'conv3.0.bias': 'layers.4.bias'
        }

        for old_key, new_key in old_to_new.items():
            if old_key in old_state_dict:
                new_state_dict[new_key].copy_(old_state_dict[old_key])

        new_model.load_state_dict(new_state_dict)
        return new_model