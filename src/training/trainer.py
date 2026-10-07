from typing import Dict, List
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import copy

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    
    model.train()
    
    total_loss = 0.0
    total_samples = 0

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
        
    avg_loss = total_loss/total_samples
    
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

class EarlyStopping:
    def __init__(
        self,
        mode: str,
        patience: int,
        min_delta: float = 0.0,
    ):
        if mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'")

        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self.best_value = None
        self.counter = 0

    def is_improvement(
        self,
        value: float,
    ) -> bool:

        if self.best_value is None:
            return True

        if self.mode == "min":

            return (
                value < self.best_value - self.min_delta
            )

        return (
            value > self.best_value + self.min_delta
        )

    def step(
        self,
        value: float,
    ) -> bool:

        if self.is_improvement(value):

            self.best_value = value

            self.counter = 0

            return False

        self.counter += 1

        return (
            self.counter >= self.patience
        )

def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    max_epochs: int,
    validation_metric_fn,
    monitor_name: str,
    monitor_mode: str,
    patience: int,
    min_delta: float = 0.0,
    restore_best_weights: bool = True,
    verbose: bool = True,
):

    history = {
        "train_loss": [],
        "val_loss": [],
        monitor_name: [],
    }

    model.to(device)

    early_stopping = EarlyStopping(
        mode=monitor_mode,
        patience=patience,
        min_delta=min_delta,
    )

    best_state = None
    best_epoch = None
    best_metric = None

    stopped_epoch = max_epochs

    for epoch in range(1, max_epochs + 1,):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss = evaluate_loss(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        metric_value = validation_metric_fn(
            model,
            val_loader,
            device,
        )

        history["train_loss"].append(train_loss)

        history["val_loss"].append(val_loss)

        history[monitor_name].append(metric_value)

        improvement = (
            early_stopping
            .is_improvement(metric_value)
        )

        if improvement:
            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch

            best_metric = metric_value

        should_stop = (
            early_stopping.step(
                metric_value
            )
        )

        if verbose:
            print(
                f"Epoch "
                f"{epoch:03d}/"
                f"{max_epochs:03d}"
                f" | "
                f"train_loss="
                f"{train_loss:.6f}"
                f" | "
                f"val_loss="
                f"{val_loss:.6f}"
                f" | "
                f"{monitor_name}="
                f"{metric_value:.6f}"
            )

        if should_stop:
            stopped_epoch = epoch
            if verbose:
                print(
                    "\nEarly stopping:"
                    f" epoch={epoch}"
                    f" | best_epoch="
                    f"{best_epoch}"
                    f" | best_"
                    f"{monitor_name}="
                    f"{best_metric:.6f}"
                )
            break
        
    if (restore_best_weights and best_state is not None):
        model.load_state_dict(best_state)

    training_info = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "stopped_epoch": stopped_epoch,
        "epochs_trained":
            len(
                history["train_loss"]
            ),
    }

    return (
        history,
        training_info,
    )