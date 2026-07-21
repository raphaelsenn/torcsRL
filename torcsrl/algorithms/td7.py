from typing import Dict, Any
from collections import deque

import copy
import time
import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

from torcsrl.buffers.lap_replay_buffer import LapReplayBuffer
from torcsrl.algorithms.base import OffPolicyAlgorithm
from torcsrl.models.td7_networks import ActorMLP, CriticMLP, EncoderMLP


class TD7(OffPolicyAlgorithm):
    def __init__(
        self,
        train_env: gym.Env,
        val_env: gym.Env,
        *,
        ac_kwargs: Dict[str, Any],
        lr_actor: float = 0.0003,
        lr_critic: float = 0.0003,
        lr_encoder: float = 0.0003,
        gamma: float = 0.992,
        lap_alpha: float = 0.4,
        lap_min_priority: float = 1.0,
        buffer_size: int = 1_000_000,
        buffer_start_size: int = 10_000,
        batch_size: int = 256,
        exploration_noise: float = 0.1,
        noise_target_network: float = 0.2,
        noise_target_clip: float = 0.5,
        gradient_steps: int = 1,
        policy_delay: int = 2,
        target_update_freq: int = 250,
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
            tau_polyak=None,
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
        self.lr_encoder = lr_encoder
        self.policy_delay = policy_delay
        self.target_update_freq = target_update_freq
        self.lap_alpha = lap_alpha
        self.lap_min_priority = lap_min_priority

        self.replay_buffer = LapReplayBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            capacity=buffer_size,
            batch_size=batch_size,
            device=device,
        )

        self.exploration_noise = exploration_noise
        self.noise_tgt_net = noise_target_network
        self.noise_tgt_clip = noise_target_clip

        self.actor = ActorMLP(obs_dim=self.obs_dim, action_space=self.action_space, **ac_kwargs).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = CriticMLP(obs_dim=self.obs_dim, action_dim=self.action_dim, **ac_kwargs).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.encoder = EncoderMLP(obs_dim=self.obs_dim, action_dim=self.action_dim, **ac_kwargs).to(self.device)
        self.encoder_fixed = copy.deepcopy(self.encoder)
        self.encoder_fixed_target = copy.deepcopy(self.encoder)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), self.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), self.lr_critic)
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), self.lr_encoder)

		# Value clipping tracked values
        self.max = -1e8
        self.min = 1e8
        self.max_target = 0
        self.min_target = 0

    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        zs = self.encoder_fixed.zs(obs) 
        action = self.actor(obs, zs)

        if deterministic is False:
            noise = self.exploration_noise * torch.randn_like(action, generator=self.torch_rng) 
            action = action + noise
            action = action.clamp(self.action_low_torch, self.action_high_torch)

        return action

    @torch.no_grad()
    def update_target_networks(self) -> None:
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # The target receives the old fixed encoder first.
        self.encoder_fixed_target.load_state_dict(self.encoder_fixed.state_dict())
        self.encoder_fixed.load_state_dict(self.encoder.state_dict())

    def n_gradient_steps(self) -> None:
        for _ in range(self.gradient_steps):
            obs, action, ret, obs_next, done = self.replay_buffer.sample()
            
            # =================== Update Encoder (f, g) =======================
            with torch.no_grad():
                zs_next = self.encoder.zs(obs_next)

            zs = self.encoder.zs(obs)
            pred_zs = self.encoder.zsa(zs, action)
            encoder_loss = F.mse_loss(pred_zs, zs_next)

            self.optimizer_encoder.zero_grad()
            encoder_loss.backward()
            self.optimizer_encoder.step()

            # =============== Compute targets for Q function ==================
            with torch.no_grad():
                # Target state representation
                fixed_target_zs = self.encoder_fixed_target.zs(obs_next) 
                
                # Target policy smoothing
                noise = self.noise_tgt_net * torch.randn_like(action, generator=self.torch_rng) 
                noise_clip = noise.clamp(-self.noise_tgt_clip, self.noise_tgt_clip)
                action_next = self.actor_target(obs_next, fixed_target_zs) + noise_clip                 # Target network
                action_next = action_next.clamp(self.action_low_torch, self.action_high_torch)

                # Target state-action representation 
                fixed_target_zsa = self.encoder_fixed_target.zsa(fixed_target_zs, action_next)

                # Clipped double Q-learning
                q1_tgt_next, q2_tgt_next = self.critic_target(obs_next, action_next, fixed_target_zsa, fixed_target_zs)
                q_tgt_next = torch.min(q1_tgt_next, q2_tgt_next).clamp(self.min, self.max)    # [B, 1]

                # n-step TD target
                td_target = ret + self.gamma * (1.0 - done) * q_tgt_next.view(-1)     # [B] 

                self.max = max(self.max, float(q_tgt_next.max()))
                self.min = min(self.min, float(q_tgt_next.min()))

                fixed_zs = self.encoder_fixed.zs(obs)
                fixed_zsa = self.encoder_fixed.zsa(fixed_zs, action)

            # ===================== Update Q function =========================
            q1, q2 = self.critic(obs, action, fixed_zsa, fixed_zs)
            q = torch.cat([q1, q2], dim=-1)     # [B, 2] 
            td_loss = (q - td_target.unsqueeze(-1)).abs()     # [B, 2] - [B, 1] = [B, 2]
            loss_q = torch.where(td_loss < self.lap_min_priority, 0.5 * td_loss.pow(2), self.lap_min_priority * td_loss).sum(1).mean()

            self.optimizer_critic.zero_grad()
            loss_q.backward()
            self.optimizer_critic.step()

            # ========================= Update LAP =============================
            # LAP was introduced in the paper: 
            #   For SALE: State-Action Representation Learning for Deep Reinforcement Learning, Fujimoto et al., 2023
            # Read more here:
            #   https://arxiv.org/abs/2306.02451
            #   https://github.com/sfujim/TD7/blob/main/TD7.py,
            priorities = td_loss.max(1)[0].clamp(min=self.lap_min_priority).pow(self.lap_alpha) 
            self.replay_buffer.update_priorities(priorities)

            # ========================= Update policy =========================
            if self.grad_step_counter % self.policy_delay == 0: 
                
                # Freeze params of Q-function (save computational effort) 
                for p in self.critic.parameters():
                    p.requires_grad = False 

                action_act = self.actor(obs, fixed_zs)
                fixed_zsa = self.encoder_fixed.zsa(fixed_zs, action_act) 
                
                q1, q2 = self.critic(obs, action_act, fixed_zsa, fixed_zs)
                q = torch.cat([q1, q2], dim=-1)
                loss_actor = torch.mean(-q)

                self.optimizer_actor.zero_grad()
                loss_actor.backward()
                self.optimizer_actor.step()

                for p in self.critic.parameters():
                    p.requires_grad = True

            # ==================== Update target networks =====================
            if self.grad_step_counter % self.target_update_freq == 0: 
                self.update_target_networks()
                self.replay_buffer.reset_max_priority()
                self.max_target = self.max
                self.min_target = self.min

            self.grad_step_counter += 1         # NOTE: INCREASE GRADIENT STEP COUNTER

    def train(self, n_timesteps: int) -> None:
        obs, info = self.collect_rollouts()

        self.episode_counter = 1
        self.grad_step_counter = 1
        self.global_step_counter = 1

        should_eval = False
        should_save = False
        start_time = time.monotonic()

        for _ in range(1, n_timesteps + 1):
            action = self.act_numpy(obs, deterministic = False)
            obs_next, reward, terminated, truncated, info = self.env.step(action)
            self.replay_buffer.push(obs, action, reward, obs_next, terminated)
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

            self.global_step_counter += 1       # NOTE: INCREASE GLOBAL STEP COUNTER

        self.val_env.close()
        self.env.close()
