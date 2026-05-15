import copy
import os
import torch
import torch.nn.functional as F

from torcsrl.config import EnvConfig
from torcsrl.algorithms.base import OffPolicyAlgorithm
from torcsrl.models.ac_td3 import ActorMLP, CriticMLP


class TD3(OffPolicyAlgorithm):
    """
    Twin Delayed Deep Deterministic Policy Gradient Algorithm (TD3).

    Reference:
    ----------
    Addressing Function Approximation Error in Actor-Critic Methods, Fujimoto et al., 2018
    https://arxiv.org/abs/1802.09477   
    
    """  
    def __init__(
        self,
        env_cfg: EnvConfig,
        *,
        lr_actor: float,
        lr_critic: float,
        gamma: float = 0.99,
        tau_polyak: float = 0.005,
        buffer_size: int = 1_000_000,
        buffer_start_size: int = 10_000,
        batch_size: int = 256,
        epsilon: float = 0.1,
        epsilon_tgt_net: float = 0.2,
        epsilon_clip: float = 0.5,
        gradient_steps: int = 1,
        policy_delay: int = 2,
        save_every: int = 1_000,
        eval_every: int = 1_000,
        n_eval_runs: int = 5,
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
        self.policy_delay = policy_delay

        self.epsilon = epsilon
        self.epsilon_tgt_net = epsilon_tgt_net
        self.epsilon_clip = epsilon_clip

        self.actor = ActorMLP(self.obs_dim, self.action_dim).to(self.device)
        self.actor_tgt = copy.deepcopy(self.actor)

        self.critic = CriticMLP(self.obs_dim, self.action_dim).to(self.device)
        self.critic_tgt = copy.deepcopy(self.critic)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), self.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), self.lr_critic)

    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        action = self.actor.act(obs)

        if deterministic is False:
            action += self.epsilon * torch.randn_like(action)

        return action

    @torch.no_grad()
    def update_target_networks(self) -> None:
        params = zip(self.actor.parameters(), self.actor_tgt.parameters())
        for theta, theta_tgt in params:
            theta_tgt.data.copy_(self.tau_polyak * theta.data + (1.0 - self.tau_polyak) * theta_tgt.data) 

        params = zip(self.critic.parameters(), self.critic_tgt.parameters())
        for theta, theta_tgt in params:
            theta_tgt.data.copy_(self.tau_polyak * theta.data + (1.0 - self.tau_polyak) * theta_tgt.data) 

    def n_gradient_steps(self, step: int) -> None:
        for _ in range(self.gradient_steps):
            obs, action, reward, obs_next, done = self.replay_buffer.sample()
            
            # =============== Compute targets for Q function ==================
            with torch.no_grad():
                # Target policy smoothing
                noise = self.epsilon_tgt_net * torch.randn_like(action) 
                noise_tgt = noise.clip(-self.epsilon_clip, self.epsilon_clip)
                action_pi_tgt_next = self.actor_tgt.act(obs_next) + noise_tgt

                # Clipped double Q-learning
                q1_tgt_next, q2_tgt_next = self.critic_tgt(obs_next, action_pi_tgt_next)
                q_tgt_next = torch.min(q1_tgt_next, q2_tgt_next).view(-1) 

                # 1-step TD target 
                td_target = reward
                td_target += self.gamma * (1.0 - done) * q_tgt_next
            
            # ===================== Update Q function =========================
            q1, q2 = self.critic(obs, action)
            
            loss_q1 = F.mse_loss(q1.view(-1), td_target)
            loss_q2 = F.mse_loss(q2.view(-1), td_target)
            loss_q = loss_q1 + loss_q2

            self.optimizer_critic.zero_grad()
            loss_q.backward()
            self.optimizer_critic.step()

            # ========================= Update policy =========================
            if step % self.policy_delay == 0: 
                
            # Freeze params of Q-function (save computational effort) 
                for p in self.critic.parameters():
                    p.requires_grad = False 
                
                action_pi = self.act(obs, deterministic=True)
                q1 = self.critic.q1(obs, action_pi).view(-1)
                loss_pi = torch.mean(-q1)

                self.optimizer_actor.zero_grad()
                loss_pi.backward()
                self.optimizer_actor.step()

                for p in self.critic.parameters():
                    p.requires_grad = True

            # ==================== Update target networks =====================
                self.update_target_networks()

    def train(self, n_timesteps: int) -> None:
        self.collect_rollouts()

        obs, _ = self.env.reset()
        episode = 0

        for step in range(1, n_timesteps + 1):
            action = self.act_numpy(obs, deterministic=False)
            obs_next, reward, terminated, truncated, _ = self.env.step(action)
            self.replay_buffer.push(obs, action, reward, obs_next, terminated)
            self.n_gradient_steps(step)
            obs = obs_next

            if terminated or truncated:
                obs, _ = self.env.reset()
                episode += 1

            if step % self.eval_every == 0:
                self.evaluate(step)
                if self.verbose:
                    self.print_stats(step, episode)

            if step % self.save_every == 0:
                self.save()

        self.evaluate(n_timesteps)
        self.save()
        self.env.close()

    def save(self) -> None:
        algo_name = self.__class__.__name__.lower()
        run_name = f"{algo_name}-lr_pi{self.lr_actor}-lr_q{self.lr_critic}-seed{self.seed}"
        save_dir = os.path.join("checkpoints", algo_name, run_name)
        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, "critic.pt"))
        self.eval_stats.to_csv(os.path.join(save_dir, "eval_stats.csv"))