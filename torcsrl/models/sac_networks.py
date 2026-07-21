import torch
import torch.nn as nn
import torch.nn.functional as F

from gymnasium.spaces import Box

from torcsrl.models.base import Actor, Critic


class ActorMLP(Actor):
    def __init__(
        self,
        obs_dim: int,
        action_space: Box,
        hidden_dim: int,
        **kw_args 
    ) -> None:
        super().__init__(obs_dim, action_space)

        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )

        self.mu = nn.Linear(hidden_dim, self.action_dim)
        self.log_std = nn.Linear(hidden_dim, self.action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs_enc = self.mlp(obs)
        mu = self.mu(obs_enc) 
        log_std = self.log_std(obs_enc)
        log_std = log_std.clamp(-20.0, 2.0)
        std = log_std.exp()
        return mu, std

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self(obs)
        return self.action_scale * F.tanh(mu) + self.action_bias


class CriticMLP(Critic):
    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_dim: int,
            **kw_args
    ) -> None:
        super().__init__(obs_dim, action_dim)

        self.Q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(True),
  
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            
            nn.Linear(hidden_dim, 1)
        )

        self.Q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(True),
        
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            
            nn.Linear(hidden_dim, 1)
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
    