from typing import Tuple
import numpy as np
import torch

from torcsrl.buffers.base import BufferBase


class NStepLapReplayBuffer(BufferBase):
    """
    Loss-Adjusted Prioritized Experience Replay.
    
    Read more here: https://arxiv.org/pdf/2007.06049 
    """
    def __init__(
            self, 
            obs_dim: int, 
            action_dim: int, 
            capacity: int, 
            batch_size: int,
            device: str = "cpu"
    ) -> None:
        super().__init__(obs_dim, action_dim, capacity, batch_size, device)
        self.steps = np.empty_like(self.dones)   # Only relevant if t + n > T
        self.priorities = np.zeros_like(self.rewards)
        self.max_priority = 1.0

    def push(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: float, 
            obs_next: np.ndarray, 
            dones: bool,
            steps: int,
    ) -> None:
        pos = self.position

        self.obs[pos] = obs.astype(np.float32)
        self.actions[pos] = action.astype(np.float32)
        self.rewards[pos] = float(reward)
        self.obs_next[pos] = obs_next.astype(np.float32)
        self.dones[pos] = float(dones)
        self.steps[pos] = float(steps)
        self.priorities[pos] = self.max_priority

        self.position = (pos + 1) % self.capacity 
        self.size = min(self.size + 1, self.capacity)

    @torch.no_grad()
    def sample(self) -> Tuple[torch.Tensor, ...]:
        assert self.size >= self.batch_size, (
            f"ReplayBuffer of size {self.size} can't sample batch of size {self.batch_size}."
        )
        batch_size = min(self.size, self.batch_size)

        csum = torch.cumsum(torch.as_tensor(self.priorities[:self.size], dtype=torch.float32, device=self.device), 0)
        val = torch.rand(size=(batch_size,), device=self.device) * csum[-1]
        self.indices = torch.searchsorted(csum, val).cpu().numpy()
        
        obs_bt = torch.as_tensor(self.obs[self.indices], dtype=torch.float32, device=self.device)
        actions_bt = torch.as_tensor(self.actions[self.indices], dtype=torch.float32, device=self.device)
        rewards_bt = torch.as_tensor(self.rewards[self.indices], dtype=torch.float32, device=self.device)
        obs_next_bt = torch.as_tensor(self.obs_next[self.indices], dtype=torch.float32, device=self.device)
        dones_bt = torch.as_tensor(self.dones[self.indices], dtype=torch.float32, device=self.device)
        steps_bt = torch.as_tensor(self.steps[self.indices], dtype=torch.float32, device=self.device)

        return obs_bt, actions_bt, rewards_bt, obs_next_bt, dones_bt, steps_bt
    
    def update_priorities(self, priorities: torch.Tensor | np.ndarray) -> None:
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.view(-1).detach().cpu().numpy()
        else:
            priorities = np.asarray(priorities).reshape(-1)

        self.priorities[self.indices] = priorities
        self.max_priority = max(float(self.priorities.max()), self.max_priority)

    def reset_max_priority(self):
        self.max_priority = float(self.priorities[:self.size].max())
