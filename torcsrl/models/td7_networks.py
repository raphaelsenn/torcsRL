import torch
import torch.nn as nn
import torch.nn.functional as F

from gymnasium.spaces import Box

from torcsrl.models.base import Actor, Critic


def AvgL1Norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x/x.abs().mean(-1, keepdim=True).clamp(min=eps)


class EncoderMLP(nn.Module):
	def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        zs_dim: int = 256, 
        hidden_dim: int = 256,
        **kw_args
    ) -> None:
		super().__init__()

		# State encoder
		self.zs1 = nn.Linear(obs_dim, hidden_dim)
		self.zs2 = nn.Linear(hidden_dim, hidden_dim)
		self.zs3 = nn.Linear(hidden_dim, zs_dim)
		
		# State-action encoder
		self.zsa1 = nn.Linear(zs_dim + action_dim, hidden_dim)
		self.zsa2 = nn.Linear(hidden_dim, hidden_dim)
		self.zsa3 = nn.Linear(hidden_dim, zs_dim)
	
	def zs(self, obs: torch.Tensor) -> torch.Tensor:
		zs = F.elu(self.zs1(obs))
		zs = F.elu(self.zs2(zs))
		zs = AvgL1Norm(self.zs3(zs))
		return zs

	def zsa(self, zs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		zsa = F.elu(self.zsa1(torch.cat([zs, action], 1)))
		zsa = F.elu(self.zsa2(zsa))
		zsa = self.zsa3(zsa)
		return zsa


class ActorMLP(Actor):
    def __init__(
        self,
        obs_dim: int,
        action_space: Box,
        hidden_dim: int,
        zs_dim: int,
        **kw_args
    ) -> None:
        super().__init__(obs_dim, action_space)

        self.l0 = nn.Linear(obs_dim, hidden_dim)
        self.l1 = nn.Linear(zs_dim + hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, self.action_dim)

    def forward(self, obs: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
        a = AvgL1Norm(self.l0(obs))
        a = torch.cat([a, zs], 1)
        a = F.relu(self.l1(a))
        a = F.relu(self.l2(a))
        return self.action_scale * F.tanh(self.l3(a)) + self.action_bias

    def act(self, obs: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
        action = self(obs, zs)
        return action


class CriticMLP(Critic):
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        zs_dim: int = 256, 
        hidden_dim: int = 256,
        **kw_args
    ) -> None:
        super().__init__(obs_dim=obs_dim, action_dim=action_dim)

        self.q01 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.q1 = nn.Linear(2 * zs_dim + hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, hidden_dim)
        self.q3 = nn.Linear(hidden_dim, 1)

        self.q02 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.q4 = nn.Linear(2 * zs_dim + hidden_dim, hidden_dim)
        self.q5 = nn.Linear(hidden_dim, hidden_dim)
        self.q6 = nn.Linear(hidden_dim, 1)

    def forward(
        self, 
        obs: torch.Tensor, 
        action: torch.Tensor,
        zsa: torch.Tensor, 
        zs: torch.Tensor
    ) -> torch.Tensor:
        sa = torch.cat([obs, action], 1)
        embeddings = torch.cat([zsa, zs], 1)

        q1 = AvgL1Norm(self.q01(sa))
        q1 = torch.cat([q1, embeddings], 1)
        q1 = F.elu(self.q1(q1))
        q1 = F.elu(self.q2(q1))
        q1 = self.q3(q1)

        q2 = AvgL1Norm(self.q02(sa))
        q2 = torch.cat([q2, embeddings], 1)
        q2 = F.elu(self.q4(q2))
        q2 = F.elu(self.q5(q2))
        q2 = self.q6(q2)
        return q1, q2
        # return torch.cat([q1, q2], 1)

    def Q1(
        self, 
        obs: torch.Tensor, 
        action: torch.Tensor,
        zsa: torch.Tensor, 
        zs: torch.Tensor
    ) -> torch.Tensor:
        sa = torch.cat([obs, action], 1)
        embeddings = torch.cat([zsa, zs], 1)

        q1 = AvgL1Norm(self.q01(sa))
        q1 = torch.cat([q1, embeddings], 1)
        q1 = F.elu(self.q1(q1))
        q1 = F.elu(self.q2(q1))
        q1 = self.q3(q1)
        
        return q1

    def Q2(
        self, 
        obs: torch.Tensor, 
        action: torch.Tensor,
        zsa: torch.Tensor, 
        zs: torch.Tensor
    ) -> torch.Tensor:
        sa = torch.cat([obs, action], 1)
        embeddings = torch.cat([zsa, zs], 1)

        q2 = AvgL1Norm(self.q02(sa))
        q2 = torch.cat([q2, embeddings], 1)
        q2 = F.elu(self.q4(q2))
        q2 = F.elu(self.q5(q2))
        q2 = self.q6(q2)
        return q2
