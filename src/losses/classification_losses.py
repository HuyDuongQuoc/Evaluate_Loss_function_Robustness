import torch
import torch.nn.functional as F
from torch import nn

class GeneralizedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        q: float=0.7,
        eps: float = 1e-7,
    ):
        super().__init__()
        
        if not 0.0 < q <=1.0:
            raise ValueError("q must satisfy (0,1]")
        
        self.q = q
        self.eps = eps
    
     
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = F.softmax(logits, dim=1)
        
        probabilities = probabilities.clamp(min=self.eps, max=1.0)
        
        target_probabilities = probabilities.gather(
            dim=1,
            index=targets.unsqueeze(1),
        ).squeeze(1)
        
        loss = (1-target_probabilities.pow(self.q))/self.q
        
        return loss.mean()
    

class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        alpha: float=0.1,
        beta: float=1.0,
        eps: float=1e-4,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    )->torch.Tensor:
        
        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
        )
        
        probabilities = F.softmax(logits, dim=1)
        probabilities = probabilities.clamp(min=1e-7, max=1.0)
        
        one_hot = F.one_hot(
            targets,
            num_classes = self.num_classes,
        ).float()
        
        one_hot.clamp(min=self.eps, max=1.0)
        
        one_hot_safe = one_hot.clamp(
            min=self.eps,
            max=1.0,
        )
        
        rce = -torch.sum(
            probabilities*torch.log(one_hot_safe),
            dim=1, 
        )
        
        loss = (self.alpha*ce + self.beta*rce)
        
        return loss.mean()
    
def get_classification_loss(
    name: str,
    num_classes: int=3,
    gce_q: float=0.7,
    sce_alpha: float=0.1,
    sce_beta: float=1.0,   
)->nn.Module:
    
    name = name.lower()
    
    if name == "ce":
        return nn.CrossEntropyLoss()
    
    if name == "gce":
        return GeneralizedCrossEntropyLoss(q=gce_q)
    
    if name == "sce":
        return SymmetricCrossEntropyLoss(
            num_classes = num_classes,
            alpha = sce_alpha,
            beta = sce_beta,
        )
        
    raise ValueError(f"Unknow classification loss: {name}")

