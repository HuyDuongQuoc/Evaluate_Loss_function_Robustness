from typing import Literal
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

tasks = Literal["regression", "classification"]

def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    task: tasks,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int=0,
) -> DataLoader:
    
    X_tensor = torch.as_tensor(
        X,
        dtype=torch.float32,
    )
    
    if task=="regression":
        y_tensor = torch.as_tensor(
            y,
            dtype=torch.float32,
        )
    
    elif task == "classification":
        y_tensor = torch.as_tensor(
            y,
            dtype = torch.long,
        )
    
    else:
        raise ValueError(f"Unknown task: {task}")
    
    dataset = TensorDataset(X_tensor,y_tensor)
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator = generator if shuffle else None,
    )
    