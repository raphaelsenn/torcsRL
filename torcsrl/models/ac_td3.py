import torch
import torch.nn as nn

from gymnasium.spaces import Box

from torcsrl.models.base import Actor, Critic


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
            
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(True),

            nn.Linear(h2_dim, self.action_dim),
            nn.Tanh()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.action_scale * self.mlp(obs) + self.action_bias


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

        self.Q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, h1_dim),
            nn.ReLU(True),
  
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(True),
            
            nn.Linear(h2_dim, 1)
        )

        self.Q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, h1_dim),
            nn.ReLU(True),
        
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(True),
            
            nn.Linear(h2_dim, 1)
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cat = torch.cat([obs, action], dim=-1) 
        return self.Q1(cat), self.Q2(cat)

    def q(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(obs, action)

    def q1(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([obs, action], dim=-1) 
        return self.Q1(cat)
    
    def q2(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([obs, action], dim=-1) 
        return self.Q2(cat)