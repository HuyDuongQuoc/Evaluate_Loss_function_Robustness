import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

@torch.no_grad()
def evaluate_regression(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    
    model.eval()
    
    predictions = []
    targets = []
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        
        outputs = model(X_batch)
        
        predictions.append(outputs.cpu().numpy())
        
        targets.append(y_batch.numpy())
        
    predictions = np.concatenate(predictions)
    
    targets = np.concatenate(targets)
    
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    
    return{
        "mae": float(mae),
        "rmse": float(rmse),
    }
        
        
@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    
    model.eval()
    predictions = []
    targets = []
    
    for X_batch, y_batch in dataloader:
        
        X_batch = X_batch.to(device)
        
        logits = model(X_batch)
        
        predicted_classes = torch.argmax(
            logits,
            dim=1,
        ) 
        
        predictions.append(predicted_classes.cpu().numpy())
        
        targets.append(y_batch.numpy())
        
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average = "macro")
    cm = confusion_matrix(targets, predictions)
    
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm,
    }