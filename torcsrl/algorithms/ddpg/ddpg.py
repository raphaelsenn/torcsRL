from typing import Dict

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torcsrl.config import EnvConfig
from torcsrl.algorithms.base.off_policy import OffPolicyAlgorithm
from torcsrl.algorithms.ddpg.actor_critic import Actor, Critic


class DDPG(OffPolicyAlgorithm):
    """
    Deep Deterministic Policy Gradient Algorithm (DDPG).

    Reference:
    ----------
    Continuous control with deep reinforcement learning, Lillicrap et al., 2015
    https://arxiv.org/abs/1509.02971
    """  
    def __init__(
        self,
        env_cfg: EnvConfig,
        *,
        lr_actor: float,
        lr_critic: float,
        gamma: float = 0.99,
        tau_polyak: float = 0.995,
        buffer_size: int = 1_000_000,
        buffer_start_size: int = 10_000,
        batch_size: int = 256,
        std_noise: float = 0.1,
        gradient_steps: int = 1,
        save_every: int = 1_000,
        eval_every: int = 1_000,
        n_eval_runs: int = 10,
        verbose: bool = True,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            env_cfg=env_cfg,
            gamma=gamma,
            tau_polyak=tau_polyak,
            buffer_size=buffer_size,
            buffer_start_size=buffer_start_size,
            batch_size=batch_size,
            gradient_steps=gradient_steps,
            save_every=save_every,
            eval_every=eval_every,
            n_eval_runs=n_eval_runs,
            verbose=verbose,
            seed=seed,
            device=device,
        )

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.std_noise = std_noise

        self.actor = Actor(self.obs_dim, self.action_dim,  std_noise=std_noise).to(self.device)
        self.actor_tgt = copy.deepcopy(self.actor)

        self.critic = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic_tgt = copy.deepcopy(self.critic)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), self.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), self.lr_critic)

    @torch.no_grad()
    def update_target_networks(self) -> None:
        params = zip(self.actor.parameters(), self.actor_tgt.parameters())
        for theta, theta_tgt in params:
            theta_tgt.data = self.tau_polyak * theta.data + (1 - self.tau_polyak) * theta_tgt.data
       
        params = zip(self.critic.parameters(), self.critic_tgt.parameters())
        for theta, theta_tgt in params:
            theta_tgt.data = self.tau_polyak * theta.data + (1 - self.tau_polyak) * theta_tgt.data

    def n_gradient_steps(self) -> None:
        for step in range(self.gradient_steps):
            obs, actions, rewards, obs_next, dones = self.replay_buffer.sample()

            with torch.no_grad():
                actions_next = self.act(obs_next, deterministic=True)
                q_tgt_next = self.critic_tgt(obs_next, actions_next).view(-1)
                td_target = self.gamma * (1.0 - dones) * q_tgt_next
                td_target += rewards

            q = self.critic(obs, actions).view(-1)
            loss_q = F.mse_loss(q, td_target)

            self.optimizer_critic.zero_grad()
            loss_q.backward()
            self.optimizer_critic.step()

            actions_pi = self.act(obs, True)
            q = self.critic(obs, actions_pi).view(-1)
            loss_pi = -torch.mean(q)

            self.optimizer_actor.zero_grad()
            loss_pi.backward()
            self.optimizer_actor.step()

            self.update_target_networks()

    def train(self, n_timesteps: int) -> None:
        self.collect_rollouts()
        obs, _ = self.env.reset(seed=self.seed)
        total_episodes = 0

        for step in range(1, n_timesteps + 1):
            action = self.act_numpy(obs, deterministic=False)
            obs_next, reward, terminated, truncated, _ = self.env.step(action)
            self.replay_buffer.push(obs, action, reward, obs_next, terminated)
            self.n_gradient_steps()
            self.periodic_tasks(step, total_episodes) 
            obs = obs_next

            if terminated or truncated:
                obs, _ = self.env.reset()
                total_episodes += 1

        self.save()
        self.env.close()

    def get_models(self) -> Dict[str, nn.Module]:
        return {
            "actor" : self.actor,
            "critic": self.critic
        }