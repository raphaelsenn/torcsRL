import torch
import torch.nn as nn


class Actor(nn.Module):
        """
        This actor implementation differs from the original paper.
        
        Reference:
        ---------- 
        Addressing Function Approximation Error in Actor-Critic Methods, Fujimoto et al., 2018
        https://arxiv.org/abs/1802.09477 
        
        Continuous control with deep reinforcement learning, 
        https://arxiv.org/abs/1509.02971, Lillicrap et al., 2015
        """ 
        def __init__(
                self, 
                obs_dim: int, 
                action_dim: int,
                h1_dim: int = 256,
                h2_dim: int = 256,
                std_noise: float = 0.1
        ) -> None:
            super().__init__()

            self.std_noise = std_noise

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

        def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
            action = self(obs)
        
            if not deterministic:
                action += self.std_noise * torch.randn_like(action)
        
            action = action.clamp(-1.0, 1.0)  

            return action


class Critic(nn.Module):
        """
        This critic implementation differs from the original paper. 
        
        Reference:
        ---------- 
        Addressing Function Approximation Error in Actor-Critic Methods, Fujimoto et al., 2018
        https://arxiv.org/abs/1802.09477  
        

        Continuous control with deep reinforcement learning, 
        https://arxiv.org/abs/1509.02971, Lillicrap et al., 2015
        """ 
        def __init__(
                self, 
                obs_dim: int, 
                action_dim: int,
                h1_dim: int = 256,
                h2_dim: int = 256,

        ) -> None:
            super().__init__()

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