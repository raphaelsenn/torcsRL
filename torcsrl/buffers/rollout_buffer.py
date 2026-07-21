from typing import Tuple

import numpy as np
import torch

from torcsrl.buffers.base import BufferBase


class RolloutBuffer(BufferBase):
    def __init__(
            self, 
            obs_dim: Tuple, 
            action_dim: Tuple, 
            horizon: int, 
            batch_size: int,
    ) -> None:
        super().__init__(obs_dim, action_dim, horizon)
        self.horizon = horizon
        self.batch_size = batch_size
        self.values = np.empty(shape=(horizon,), dtype=np.float32)
        self.log_probs = np.empty(shape=(horizon,), dtype=np.float32)
        self.advantages = np.empty(shape=(horizon,), dtype=np.float32)
        self.td_errors = np.empty(shape=(horizon,), dtype=np.float32)
        self.obs_next = None

    def compute_rtgs(self, gamma: float) -> None:
        rtg = 0.0
        for t in reversed(range(self.horizon)): 
            not_done = 1.0 - self.dones[t] 
            reward = self.rewards[t]
            rtg = reward + gamma * not_done * rtg
            self.rtgs[t] = rtg

    def compute_advantages(self, gamma: float, gae_lambda: float) -> None:
        adv = 0.0
        for t in reversed(range(self.horizon)): 
            not_done = 1.0 - self.dones[t]
            reward = self.rewards[t]
            value = self.values[t]
            value_nxt = self.values[t + 1]

            td_error = reward + gamma * not_done * value_nxt - value
            adv = td_error + gamma * gae_lambda * not_done * adv
            self.advantages[t] = adv

    def push(self, s: np.ndarray, a: np.ndarray, r: float, log_prob: float, value: float, done: bool) -> None:
        i = self.size 
        horizon = self.horizon
        
        self.obs[i] = s.astype(np.float32)
        self.actions[i] = a.astype(np.float32)
        self.rewards[i] = float(r)
        self.dones[i] = float(done)
        
        self.log_probs[i] = float(log_prob)
        self.values[i] = float(value)

        self.position = (self.position + 1) % horizon

    def sample(self) -> Tuple[torch.Tensor, ...]:
        assert self.size >= self.batch_size, (
            f"ReplayBuffer of size {self.size} can't sample batch of size {self.batch_size}."
        )
        batch_size = min(self.size, self.batch_size)
        indices = np.random.randint(0, self.size, size=batch_size, dtype=np.int64)

        obs_bt = torch.as_tensor(self.obs[indices], dtype=torch.float32, device=self.device)
        actions_bt = torch.as_tensor(self.actions[indices], dtype=torch.float32, device=self.device)
        rewards_bt = torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device)
        dones_bt = torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device)
        log_probs_bt = torch.as_tensor(self.log_probs[indices], dtype=torch.float32, device=self.device)
        advantages_bt = torch.as_tensor(self.advantages[indices], dtype=torch.float32, device=self.device)

        return obs_bt, actions_bt, rewards_bt, dones_bt, log_probs_bt, advantages_bt

    def minibatches(self):
        obs, actions, log_probs, advantages, rtgs = self._flatten()
        
        if self.norm_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        idx = np.random.permutation(self.T * self.N)
        for start in range(0, self.T * self.N, self.batch_size):
            mb = idx[start:start + self.batch_size]

            yield (
                torch.as_tensor(obs[mb], dtype=torch.float32, device=self.device),              # [batch_size, obs_dim]
                torch.as_tensor(actions[mb], dtype=torch.float32, device=self.device),          # [batch_size, action_dim]
                torch.as_tensor(log_probs[mb], dtype=torch.float32, device=self.device),        # [batch_size]
                torch.as_tensor(advantages[mb], dtype=torch.float32, device=self.device),       # [batch_size]
                torch.as_tensor(rtgs[mb], dtype=torch.float32, device=self.device),             # [batch_size]
            )
