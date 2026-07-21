import torch
import torch.nn as nn
from gymnasium.spaces import Box

from torcsrl.models.base import Actor, Critic


DEFAULT_AC_KWARGS = {
    "h1_dim" : 256,
    "h2_dim" : 256,
}


class ActorMLP(Actor):
    def __init__(
        self, 
        obs_dim: int, 
        action_space: Box,
        h1_dim: int = 256,
        h2_dim: int = 256,
    ) -> None:
        super().__init__(obs_dim, action_space)

        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, h1_dim),
            nn.ReLU(True),

            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(True),

            nn.Linear(h2_dim, self.action_dim),
            nn.Tanh()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self(obs)


class CriticMLP(Critic):
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int,
        h1_dim: int = 256,
        h2_dim: int = 256,
    ) -> None:
        super().__init__(obs_dim, action_dim)

        self.h1_dim = h1_dim
        self.h2_dim = h2_dim

        self.mlp = nn.Sequential(
            nn.Linear(obs_dim + action_dim, h1_dim),
            nn.ReLU(True),

            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(True),

            nn.Linear(h2_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.mlp(x)

    def q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.forward(obs, action)
