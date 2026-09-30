from typing import Sequence

import torch
from torch import nn

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
    ):
        super().__init__()
        
        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim,hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(previous_dim, output_dim)
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
    
class RegressionMLP(MLP):
    def __init__(
        self,
        input_dim: int = 20,
        hidden_dims: Sequence[int] = (64,32),
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dims = hidden_dims,
            output_dim=1,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super().forward(x)
        return output.squeeze(-1)
    
class ClassificationMLP(MLP):
    def __init__(
        self,
        input_dim: int = 20,
        hidden_dims: Sequence[int] = (64,32),
        num_classes: int = 3,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dims = hidden_dims,
            output_dim=3,
        )