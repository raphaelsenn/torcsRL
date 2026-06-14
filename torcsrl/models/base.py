from abc import ABC, abstractmethod

import numpy as np
from gymnasium.spaces import Box

import torch
import torch.nn as nn

class Actor(nn.Module, ABC):
    def __init__(
            self, 
            obs_dim: int, 
            action_space: Box 
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = int(np.prod(action_space.shape))

        action_low_np = np.asarray(action_space.low, dtype=np.float32)
        action_high_np = np.asanyarray(action_space.high, dtype=np.float32)
        action_scale_np = (action_high_np - action_low_np) / 2.0
        action_bias_np = (action_high_np + action_low_np) / 2.0
        
        self.register_buffer("action_scale", torch.as_tensor(action_scale_np, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias_np, dtype=torch.float32))

    @abstractmethod
    def forward(self, obs: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Critic(nn.Module, ABC):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    @abstractmethod
    def forward(self, obs: torch.Tensor, actoin: torch.Tensor):
        raise NotImplementedError

    @abstractmethod 
    def q(self, obs: torch.Tensor, action: torch.Tensor):
        raise NotImplementedError