import torch
import numpy as np

def to_batched_tensor(arr: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    arr_t = torch.as_tensor(arr, dtype=torch.float32, device=device)
    arr_t = arr_t.unsqueeze(0)
    return arr_t