import json 
from pathlib import Path
from typing import Any
import numpy as np
import torch
import yaml

def load_yaml(path:str) -> dict:
    with open(path,"r",encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def make_json_serializable(obj: Any) -> Any:
    if isinstance(obj,dict):
        return {
            key: make_json_serializable(value) for key, value in obj.items()
        }
        
    if isinstance(obj,tuple):
        return [
            make_json_serializable(value) for value in obj
        ]
        
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    if isinstance(
        obj,
        (np.integer, np.floating),
    ):
        return obj.item()
    
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu()
        if obj.numel()==1:
            return obj.item()
        return obj.tolist()
    
    if isinstance(obj, Path):
        return str(obj)
    
    return obj

def save_json(
    data:dict,
    path: str|Path,
) -> None:
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = make_json_serializable(data)
    
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
        )
        
def save_checkpoint(
    model: torch.nn.Module,
    metadata:str,
    path: str|Path,
) -> None:
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
                
            "metadata":
                metadata,
        },
        path,
    )