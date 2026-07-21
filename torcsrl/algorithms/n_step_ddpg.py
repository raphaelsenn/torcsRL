from typing import Dict, Any
from collections import deque

import copy
import time
import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

from torcsrl.buffers.n_step_replay_buffer import NStepReplayBuffer
from torcsrl.algorithms.base import OffPolicyAlgorithm
from torcsrl.models.ddpg_networks import ActorMLP, CriticMLP, DEFAULT_AC_KWARGS


class NStepDDPG(OffPolicyAlgorithm):
    """
    N-Step Deep Deterministic Policy Gradient Algorithm (DDPG).

    Reference:
    ----------
    Continuous control with deep reinforcement learning, Lillicrap et al., 2015
    https://arxiv.org/abs/1509.02971
    """  
    def __init__(
        self,
        train_env: gym.Env,
        val_env: gym.Env, 
        *,
        lr_actor: float,
        lr_critic: float,
        ac_kwargs: Dict[str, Any] = DEFAULT_AC_KWARGS,
        n_steps: int = 3,
        gamma: float = 0.992,
        tau_polyak: float = 0.005,
        buffer_size: int = 3_000_000,
        buffer_start_size: int = 25_000,
        batch_size: int = 256,
        exploration_noise: float = 0.1,
        gradient_steps: int = 1,
        policy_delay: int = 2,
        save_every: int = 10_000,
        eval_every: int = 10_000,
        verbose: bool = True,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            train_env=train_env,
            val_env=val_env,
            gamma=gamma,
            ac_kwargs=ac_kwargs,
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
        
        self.n_steps = n_steps
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise

        self.replay_buffer = NStepReplayBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            capacity=buffer_size,
            batch_size=batch_size,
            device=device,
        )

        self.actor = ActorMLP(self.obs_dim, self.action_space, **self.ac_kwargs).to(self.device)
        self.actor_tgt = copy.deepcopy(self.actor)

        self.critic = CriticMLP(self.obs_dim, self.action_dim, **self.ac_kwargs).to(self.device)
        self.critic_tgt = copy.deepcopy(self.critic)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), self.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), self.lr_critic)

    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        action = self.actor.act(obs)
        
        if deterministic is False:
            noise = self.exploration_noise * torch.randn_like(action, generator=self.torch_rng)
            action = (action + noise).clamp(self.action_low_torch, self.action_high_torch)

        return action

    @torch.no_grad()
    def update_target_networks(self) -> None:
        params = zip(self.actor.parameters(), self.actor_tgt.parameters())
        for theta, theta_tgt in params:
            theta_tgt.data.copy_(self.tau_polyak * theta.data + (1.0 - self.tau_polyak) * theta_tgt.data) 

        params = zip(self.critic.parameters(), self.critic_tgt.parameters())
        for theta, theta_tgt in params:
            theta_tgt.data.copy_(self.tau_polyak * theta.data + (1.0 - self.tau_polyak) * theta_tgt.data) 

    def n_gradient_steps(self) -> None:
        for _ in range(self.gradient_steps):
            obs, action, reward, obs_next, done, n_steps = self.replay_buffer.sample()
            
            # =============== Compute targets for Q function ==================
            with torch.no_grad():
                action_next = self.actor_tgt.act(obs_next)                       # Target network
                q_tgt_next = self.critic_tgt(obs_next, action_next).view(-1)     # Target network
                td_target = reward  # R_{t + 1} + gamma*R_{t + 2} + gamma^{2}*R_{t + 3}, ... until t + n - 1
                td_target += (self.gamma**n_steps) * (1.0 - done) * q_tgt_next
            
            # ===================== Update Q function =========================
            q = self.critic(obs, action).view(-1)
            loss_q = F.mse_loss(q, td_target)

            self.optimizer_critic.zero_grad()
            loss_q.backward()
            self.optimizer_critic.step()

            # ========================= Update policy =========================
            # NOTE: Policy delay is not part of the original DDPG algorithm,
            # however, it really stabilizes training ;) (deactivate using policy_delay = 1)
            if self.grad_step_counter % self.policy_delay == 0:

                # Freeze params of Q-function (save computational effort) 
                for p in self.critic.parameters():
                    p.requires_grad = False 
                
                action_act = self.act(obs, deterministic=True)
                q = self.critic(obs, action_act).view(-1)
                loss_actor = torch.mean(-q)

                self.optimizer_actor.zero_grad()
                loss_actor.backward()
                self.optimizer_actor.step()

                for p in self.critic.parameters():
                    p.requires_grad = True

            # ==================== Update target networks =====================
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

        for _ in range(0, n_timesteps):
            # Choose action and step environment. 
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

            self.n_gradient_steps()             # NOTE: Gradient update
            obs = obs_next                      # NOTE: Obs update
            
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

        self.replay_buffer.push(s, a, ret, s_next, float(terminated), n_steps)

        r_queue.popleft()

    def collect_n_step_rollouts(self) -> tuple[np.ndarray, Dict[str, Any]]:
        env = self.env      # NOTE: Training environment
        env.action_space.seed(self.seed)
        obs, info = env.reset(seed=self.seed)

        s_queue = deque([obs], maxlen=self.n_steps + 1)
        a_queue = deque([], maxlen=self.n_steps)
        r_queue = deque([], maxlen=self.n_steps)

        for _ in range(self.buffer_start_size):
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
