from torch import nn

def get_regression_loss(
    name: str,
    huber_delta: float=1.0,
) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    
    if name == "mae":
        return nn.L1Loss()
    
    if name == "huber":
        return nn.HuberLoss(
            delta = huber_delta
        )
        
    raise ValueError(f"Unknown regression loss: {name}")

