from typing import Dict, Any
from collections import deque

import copy
import time
import torch
import torch.nn.functional as F
from torch.distributions import Normal

import gymnasium as gym
import numpy as np

from torcsrl.buffers.n_step_replay_buffer import NStepReplayBuffer
from torcsrl.algorithms.base import OffPolicyAlgorithm
from torcsrl.models.sac_networks import ActorMLP, CriticMLP


class NStepSAC(OffPolicyAlgorithm):
    def __init__(
        self,
        train_env: gym.Env,
        val_env: gym.Env,
        *,
        ac_kwargs: Dict[str, Any],
        lr_actor: float = 0.0003,
        lr_critic: float = 0.0003,
        lr_alpha: float = 0.0003,
        n_steps: int = 3,
        gamma: float = 0.992,
        tau_polyak: float = 0.005,
        buffer_size: int = 1_000_000,
        buffer_start_size: int = 10_000,
        batch_size: int = 256,
        gradient_steps: int = 1,
        target_update_interval: int = 1,
        save_every: int = 10_000,
        eval_every: int = 10_000,
        verbose: bool = True,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            train_env=train_env,
            val_env=val_env,
            ac_kwargs=ac_kwargs,
            gamma=gamma,
            tau_polyak=tau_polyak,
            buffer_size=buffer_size,
            buffer_start_size=buffer_start_size,
            batch_size=batch_size,
            gradient_steps=gradient_steps,
            save_every=save_every,
            eval_every=eval_every,
            verbose=verbose,
            seed=seed,
            device=device,
        )

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_alpha = lr_alpha
        self.n_steps = n_steps
        self.target_update_interval = target_update_interval

        self.replay_buffer = NStepReplayBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            capacity=buffer_size,
            batch_size=batch_size,
            device=device,
        )

        self.actor = ActorMLP(self.obs_dim, self.action_space, **ac_kwargs).to(self.device)
        # NOTE: SAC does not have an actor target network like DDPG or TD3

        self.critic = CriticMLP(self.obs_dim, self.action_dim, **ac_kwargs).to(self.device)
        self.critic_tgt = copy.deepcopy(self.critic)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), self.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), self.lr_critic)

        self.target_entropy = -float(self.action_dim)
        self.log_alpha = torch.nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32, device=self.device)) # exp(log(1)) = exp(0) = 1
        self.optimizer_alpha = torch.optim.Adam([self.log_alpha], self.lr_alpha)
    
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        u, std = self.actor(obs)

        if deterministic is False:
            dist = Normal(u, std)
            u = dist.rsample() # u + std * N(0, 1)

        action = self.action_scale_torch * torch.tanh(u) + self.action_bias_torch
        action = action.clamp(self.action_low_torch, self.action_high_torch)

        return action

    @torch.no_grad()
    def update_target_networks(self) -> None:
        params = zip(self.critic.parameters(), self.critic_tgt.parameters())
        for theta, theta_tgt in params:
            theta_tgt.data.copy_(self.tau_polyak * theta.data + (1.0 - self.tau_polyak) * theta_tgt.data) 

    def n_gradient_steps(self) -> None:
        # Read more about n-step SAC here:
        # SACn: Soft Actor-Critic with n-step Returns, Lyskawa et 
        # https://arxiv.org/abs/2512.13165

        for _ in range(self.gradient_steps):
            obs, action, reward_return, obs_next, done, n_steps = self.replay_buffer.sample()

            # =============== Compute targets for Q function ==================
            alpha = self.log_alpha.exp().detach()
            with torch.no_grad():
                # Sample action from policy 
                mu, std = self.actor(obs_next) 
                dist = Normal(mu, std)
                u_next = dist.rsample()
                action_next = self.action_scale_torch * F.tanh(u_next) + self.action_bias_torch

                # Log-prob correction (sum over action dim; action_dim independent (prod log => sum log))
                log_prob_next = dist.log_prob(u_next).sum(dim=-1)
                log_prob_next -= torch.log(self.action_scale_torch).sum(dim=-1)
                log_prob_next -= (2 * (np.log(2) - u_next - F.softplus(-2 * u_next))).sum(dim=-1)

                # Clipped double Q-learning
                q1_tgt_next, q2_tgt_next = self.critic_tgt(obs_next, action_next)   # Target network
                q_tgt_next = torch.min(q1_tgt_next, q2_tgt_next).view(-1) - alpha * log_prob_next

                # n-step TD target 
                td_target = reward_return + (self.gamma**n_steps) * (1.0 - done) * q_tgt_next
 
            # ===================== Update Q function =========================
            q1, q2 = self.critic(obs, action)
            loss_q1 = F.mse_loss(q1.view(-1), td_target)
            loss_q2 = F.mse_loss(q2.view(-1), td_target)
            loss_q = loss_q1 + loss_q2

            self.optimizer_critic.zero_grad()
            loss_q.backward()
            self.optimizer_critic.step()

            # ========================= Update policy =========================
                
            # Freeze params of Q-function (save computational effort) 
            for p in self.critic.parameters():
                p.requires_grad = False 
            
            # Reparemetrization trick
            # Action should be differentiable w.r.t. actor params.
            mu, std = self.actor(obs)
            dist = Normal(mu, std)
            u = dist.rsample()
            action_act = self.action_scale_torch * F.tanh(u) + self.action_bias_torch

            # Log-prob correction
            log_prob = dist.log_prob(u).sum(dim=-1)
            log_prob -= torch.log(self.action_scale_torch).sum(dim=-1)
            log_prob -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(dim=-1)

            # Actor loss 
            q1, q2 = self.critic(obs, action_act)
            q = torch.min(q1, q2).view(-1) - alpha * log_prob
            loss_actor = -q.mean()

            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            self.optimizer_actor.step()
        
            # ========================= Update alpha =========================
            loss_alpha = self.log_alpha * (-log_prob.detach() - self.target_entropy).mean()
            
            self.optimizer_alpha.zero_grad()
            loss_alpha.backward()
            self.optimizer_alpha.step()

            for p in self.critic.parameters():
                p.requires_grad = True
            
            # ==================== Update target networks =====================
            if self.grad_step_counter % self.target_update_interval == 0: 
                self.update_target_networks()

            self.grad_step_counter += 1         # NOTE: INCREASE GRADIENT STEP COUNTER

    def train(self, n_timesteps: int) -> None:
        obs, info, (s_queue, a_queue, r_queue) = self.collect_n_step_rollouts()

        self.episode_counter = 1
        self.grad_step_counter = 1
        self.global_step_counter = 1

        should_eval = False
        should_save = False
        start_time = time.monotonic()

        for _ in range(1, n_timesteps + 1):
            action = self.act_numpy(obs, deterministic=False) 
            obs_next, reward, terminated, truncated, info = self.env.step(action)

            # Update queues
            s_queue.append(obs_next)
            a_queue.append(action)
            r_queue.append(reward)

            if len(s_queue) == self.n_steps + 1:
                self.push_n_step_transition(s_queue, a_queue, r_queue, terminated)

            if terminated or truncated:
                while len(a_queue) > 0:
                    self.push_n_step_transition(s_queue, a_queue, r_queue, terminated)

            self.n_gradient_steps()
            obs = obs_next
            
            if self.global_step_counter % self.eval_every == 0:
                should_eval = True

            if self.global_step_counter % self.save_every == 0:
                should_save = True

            if terminated or truncated:
                should_eval, should_save, start_time = self.on_episode_end(should_eval, should_save, start_time)
                self.episode_counter += 1       # NOTE: INCREASE EPISODE STEP COUNTER
                obs, info = self.env.reset()

                # Clear queue. 
                s_queue.clear(); a_queue.clear(); r_queue.clear()
                s_queue.append(obs)

            self.global_step_counter += 1       # NOTE: INCREASE GLOBAL STEP COUNTER

        self.val_env.close()
        self.env.close()

    def push_n_step_transition(self, s_queue, a_queue, r_queue, terminated: bool):
        n_steps = len(r_queue) 
        
        s = s_queue.popleft()
        a = a_queue.popleft()

        ret = sum(float(r) * (self.gamma ** i) for i, r in enumerate(r_queue))
        s_next = s_queue[-1]

        self.replay_buffer.push(s, a, ret, s_next, float(terminated), float(n_steps))
        
        r_queue.popleft()
    
    def collect_n_step_rollouts(self) -> tuple[np.ndarray, Dict[str, Any]]:
        env = self.env  # NOTE: Training environment
        env.action_space.seed(self.seed)
        obs, info = env.reset(seed=self.seed)

        s_queue = deque([obs], maxlen=self.n_steps + 1)
        a_queue = deque([], maxlen=self.n_steps)
        r_queue = deque([], maxlen=self.n_steps)

        for _ in range(self.buffer_start_size):
            # Sample random action and step environment.
            action = env.action_space.sample()
            obs_next, reward, terminated, truncated, info = env.step(action)

            # Update queues
            s_queue.append(obs_next)
            a_queue.append(action)
            r_queue.append(reward)

            # Push on replay buffer
            if len(s_queue) == self.n_steps + 1:
                self.push_n_step_transition(s_queue, a_queue, r_queue, terminated)

            if terminated or truncated:
                while len(a_queue) > 0:
                    self.push_n_step_transition(s_queue, a_queue, r_queue, terminated)       

            obs = obs_next  # NOTE: Set obs to next observation!!!

            if terminated or truncated:
                obs, info = env.reset()
                s_queue.clear(); a_queue.clear(); r_queue.clear()
                s_queue.append(obs)

        return obs, info, (s_queue, a_queue, r_queue)
