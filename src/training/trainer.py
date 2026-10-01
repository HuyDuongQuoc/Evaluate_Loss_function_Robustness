from typing import Dict, List
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    
    model.train()
    
    total_loss = 0.0
    total_sample = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad(set_to_none=True) #clear old gradient
        
        outputs = model(X_batch)
        
        loss = criterion(outputs, y_batch)
        
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss detected: {loss.item()}")
        
        loss.backward()
        
        optimizer.step()
        
        batch_size = X_batch.size(0)
        
        total_loss += loss.item()*batch_size
        
        total_samples += batch_size
        
    avg_loss = total_loss/total_sample
    
    return avg_loss    

@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    
    model.eval()
    
    total_loss = 0.0
    total_sample = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        outputs = model(X_batch)
        
        loss = criterion(outputs, y_batch)
        
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite validation loss detected: {loss.item()}")
    
        batch_size = X_batch.size(0)
        
        total_loss += loss.item()*batch_size
        
        total_sample += batch_size
        
    avg_loss = total_loss/total_sample
    
    return avg_loss

def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    
    history = {
        "train_loss" : [],
        "val_loss" : [],
    }
    
    model.to(device)
    
    for epoch in range(1,epochs+1):
        train_loss = train_one_epoch(
                model=model,
                dataloader = train_loader,
                criterion=criterion,
                optimizer=optimizer,
                devic=device,
            )
        
        val_loss = evaluate_loss(
                model=model,
                dataloader = val_loader,
                criterion=criterion,
                devic=device,
            )
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        if verbose:
            print(
                f"Epoch {epoch:03d}/{epochs:03d} "
                f"| train_loss={train_loss:.6f} "
                f"| val_loss={val_loss:.6f}"
            )
            
    return history