from typing import Tuple
import numpy as np
import torch

from torcsrl.buffers.base import BufferBase


class ReplayBuffer(BufferBase):
    def __init__(
            self, 
            obs_dim: Tuple, 
            action_dim: Tuple, 
            capacity: int, 
            batch_size: int,
            device: str = "cpu"
    ) -> None:
        super().__init__(obs_dim, action_dim, capacity)
        self.batch_size = batch_size

        assert device in {"cpu", "cuda", "mps"}, (
            f"Invalid device, expected: `cpu`, `cuda` or `mps`, got: {device}."
        )
        self.device = torch.device(device)

    def sample(self) -> Tuple[torch.Tensor, ...]:
        assert self.size >= self.batch_size, (
            f"ReplayBuffer of size {self.size} can't sample batch of size {self.batch_size}."
        )
        indices = np.random.randint(0, self.size, size=self.batch_size, dtype=np.int64)

        obs_bt = torch.as_tensor(self.obs[indices], dtype=torch.float32, device=self.device)
        actions_bt = torch.as_tensor(self.actions[indices], dtype=torch.float32, device=self.device)
        rewards_bt = torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device)
        obs_next_bt = torch.as_tensor(self.obs_next[indices], dtype=torch.float32, device=self.device)
        dones_bt = torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device)

        return obs_bt, actions_bt, rewards_bt, obs_next_bt, dones_bt