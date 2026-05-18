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

        # self._init_weights()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self(obs)

    def _init_weights(self) -> None:
        # Read more here in appendix: https://arxiv.org/abs/1509.02971 
        layers = [m for m in self.mlp if isinstance(m, nn.Linear)] 
        
        for layer in layers[:-1]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        
        out = layers[-1]
        nn.init.uniform_(out.weight, -3e-3, 3e-3)
        nn.init.uniform_(out.bias, -3e-3, 3e-3)


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
            nn.Linear(h2_dim, 1),
        )

        # self._init_weights()

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.mlp(x)

    def q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.forward(obs, action)

    def _init_weights(self) -> None:
        # Read more here in appendix: https://arxiv.org/abs/1509.02971 
        layers = [m for m in self.mlp if isinstance(m, nn.Linear)]

        for layer in layers[:-1]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

        out = layers[-1]
        nn.init.uniform_(out.weight, -3e-4, 3e-4)
        nn.init.uniform_(out.bias, -3e-4, 3e-4)