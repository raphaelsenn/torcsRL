from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

from torcsrl.config import EnvConfig
from torcsrl.utils.evaluation_stats import EvluationStats
from torcsrl.utils.utils import to_batched_tensor


class RLAlgorithm(ABC):
    def __init__(
        self,
        env_cfg: EnvConfig,
        *,
        gamma: float,
        tau_polyak: float,
        save_every: int, 
        eval_every: int, 
        n_eval_runs: int = 5,
        verbose: bool = True,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        ...
        super().__init__()

        self.env_cfg = env_cfg
        self.env = gym.make(
            "TorcsSCR-v0",
            render_mode=None,
            executable=env_cfg.executable,
            port=env_cfg.port_train,
            track_name=env_cfg.track_train,
            track_category=env_cfg.track_category,
            debug=True,
        )

        assert device in {"cpu", "cuda", "mps"}, (
            f"Unkdown device, expected device in [`cpu`, `cuda`, `mps`], got: {device}."
        )
        self.device = torch.device(device)

        self.actor = ...
        self.critic = ...

        self.obs_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape
        self.obs_dim = int(np.prod(self.obs_shape))
        self.action_dim = int(np.prod(self.action_shape))

        self.action_low_np = np.asarray(self.env.action_space.low, dtype=np.float32)
        self.action_high_np = np.asanyarray(self.env.action_space.high, dtype=np.float32)
        self.action_scale_np = (self.action_high_np - self.action_low_np) / 2.0
        self.action_bias_np = (self.action_high_np + self.action_low_np) / 2.0
        
        self.action_low_torch = torch.as_tensor(self.action_low_np, dtype=torch.float32, device=self.device)
        self.action_high_torch = torch.as_tensor(self.action_high_np, dtype=torch.float32, device=self.device)
        self.action_scale_torch = torch.as_tensor(self.action_scale_np, dtype=torch.float32, device=self.device)
        self.action_bias_torch = torch.as_tensor(self.action_bias_np, dtype=torch.float32, device=self.device)

        self.gamma = gamma
        self.tau_polyak = tau_polyak
        self.verbose = verbose
        self.seed = seed
        self.save_every = save_every
        self.eval_every = eval_every 
        self.n_eval_runs = n_eval_runs
        self.eval_stats = EvluationStats()
        self.set_seeds()

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def get_models(self) -> Dict[str, nn.Module]:
        raise NotImplementedError

    @abstractmethod
    def collect_rollouts(self) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def act_numpy(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_t = to_batched_tensor(obs, self.device)
        action = self.actor.act(obs_t, deterministic)
        action = action.flatten().cpu().numpy()
        return action

    def evaluate(self, step: int) -> None:
        env = gym.make(
            "TorcsSCR-v0",
            render_mode=None,
            executable=self.env_cfg.executable,
            port=self.env_cfg.port_val,
            track_name=self.env_cfg.track_val,
            track_category=self.env_cfg.track_category,
            debug=True,
        )
        rewards = np.zeros(self.n_eval_runs)
        for episode in range(self.n_eval_runs):
            obs, _ = env.reset(seed=self.seed + episode + 1)
            env.action_space.seed(self.seed + episode + 1) 
            done = False

            while not done:
                action = self.act_numpy(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated 
                rewards[episode] += reward
        self.eval_stats.update(step, rewards)

    def set_seeds(self) -> None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def print_stats(self, step: int, episode: int) -> None:
        report = (
            f"Timestep: {step:>9d}  "
            f"Episode: {episode:>6d}  "
            f"Average Return: {self.eval_stats.last_average_reward:>10.4f}  "
            f"Std Return: {self.eval_stats.last_std_reward:>10.4f}  "
        )
        print(report)

    def periodic_tasks(self, step: int, episode: int) -> None:
        if step % self.eval_every == 0:
            self.evaluate(step)
            self.print_stats(step, episode)

        if step % self.save_every == 0:
            self.save()

    def save(self) -> None:
        name = (
            f"{self.__class__.__name__}-"
            f"Torcs-"
            f"{self.seed}.pt" 
        )
        state_dicts = {
            name: model.state_dict()
            for name, model in self.get_models().items()
        }
        torch.save(state_dicts, name)
        self.eval_stats.to_csv(name.replace(".pt", ".csv"))