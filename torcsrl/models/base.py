from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Actor(nn.Module, ABC):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

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