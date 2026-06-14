from typing import Tuple
import numpy as np
import torch

from torcsrl.buffers.base import BufferBase


class BCReplayBuffer(BufferBase):
    def __init__(
            self, 
            obs_dim: int, 
            action_dim: int, 
            capacity: int, 
            batch_size: int,
            device: str = "cpu"
    ) -> None:
        super().__init__(obs_dim, action_dim, capacity, batch_size, device)
        self.expert_actions = np.empty_like(self.actions)

    def push(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            expert_action: np.ndarray,
            reward: float, 
            obs_next: np.ndarray,
            dones: bool,
    ) -> None:
        pos = self.position

        self.obs[pos] = obs.astype(np.float32)
        self.actions[pos] = action.astype(np.float32)
        self.expert_actions[pos] = expert_action.astype(np.float32)
        self.rewards[pos] = float(reward)
        self.obs_next[pos] = obs_next.astype(np.float32)
        self.dones[pos] = float(dones)

        self.position = (pos + 1) % self.capacity 
        self.size = min(self.size + 1, self.capacity)

    def sample(self) -> Tuple[torch.Tensor, ...]:
        assert self.size >= self.batch_size, (
            f"ReplayBuffer of size {self.size} can't sample batch of size {self.batch_size}."
        )
        batch_size = min(self.size, self.batch_size)
        indices = np.random.randint(0, self.size, size=batch_size, dtype=np.int64)

        obs_bt = torch.as_tensor(self.obs[indices], dtype=torch.float32, device=self.device)
        actions_bt = torch.as_tensor(self.actions[indices], dtype=torch.float32, device=self.device)
        expert_actions_bt = torch.as_tensor(self.expert_actions[indices], dtype=torch.float32, device=self.device)
        rewards_bt = torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device)
        obs_next_bt = torch.as_tensor(self.obs_next[indices], dtype=torch.float32, device=self.device)
        dones_bt = torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device)

        return obs_bt, actions_bt, expert_actions_bt, rewards_bt, obs_next_bt, dones_bt