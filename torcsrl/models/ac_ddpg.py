import torch
import torch.nn as nn

from torcsrl.models.base import Actor, Critic


class ActorMLP(Actor):
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
            # [B, obs_dim] -> [B, h1_dim] 
            nn.Linear(obs_dim, h1_dim),
            nn.ReLU(True),

            # [B, h1_dim] -> [B, h2_dim] 
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(True),
        
            # [B, h2_dim] -> [B, action_dim] 
            nn.Linear(h2_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        action = self(obs)
        return action


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
            # [B, obs_dim + action_dim] -> [B, h1_dim] 
            nn.Linear(obs_dim + action_dim, h1_dim),
            nn.ReLU(True),
        
            # [B, h1_dim] -> [B, h2_dim] 
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(True),
        
            # [B, h2_dim] -> [B, 1] 
            nn.Linear(h2_dim, 1)
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([obs, action], dim=-1) 
        return self.mlp(cat)

    def q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.forward(obs, action)