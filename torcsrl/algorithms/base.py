from abc import ABC, abstractmethod

import random
import torch
import numpy as np
import gymnasium as gym

from torcsrl.models.base import Actor, Critic
from torcsrl.evaluation.evaluation_stats import EvluationStats
from torcsrl.buffers.replay_buffer import ReplayBuffer
from torcsrl.buffers.rollout_buffer import RolloutBuffer
from torcsrl.utils.utils import to_batched_tensor


class RLAlgorithm(ABC):
    def __init__(
        self,
        train_env: gym.Env,
        val_env: gym.Env,
        *,
        gamma: float,
        tau_polyak: float,
        save_every: int, 
        eval_every: int, 
        n_eval_runs: int,
        verbose: bool = True,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        ...
        super().__init__()

        self.env = train_env
        self.val_env = val_env

        assert device in {"cpu", "cuda", "mps"}, (
            f"Unkdown device, expected device in [`cpu`, `cuda`, `mps`], got: {device}."
        )
        self.device = torch.device(device)

        self.actor: Actor = ... 
        self.critic: Critic = ...

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
        self.save_every = save_every
        self.eval_every = eval_every 
        self.n_eval_runs = n_eval_runs
        self.eval_stats = EvluationStats()

        self.seed = seed 
        self.torch_rng = torch.Generator(device=self.device)
        self.set_seeds()

    @abstractmethod
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def collect_rollouts(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def act_numpy(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_t = to_batched_tensor(obs, self.device)
        action = self.act(obs_t, deterministic)
        action = action.flatten().cpu().numpy()
        return action

    @torch.no_grad()
    def evaluate(self, step: int) -> None:
        env = self.val_env

        rewards = np.zeros(self.n_eval_runs, dtype=np.float32)
        distance = np.zeros(self.n_eval_runs, dtype=np.float32)
        time_alive = np.zeros(self.n_eval_runs, dtype=np.float32)
        mean_speed = np.zeros(self.n_eval_runs, dtype=np.float32)
        successful = np.zeros(self.n_eval_runs, dtype=np.uint8)
        damage = np.zeros(self.n_eval_runs, dtype=np.float32)

        for episode in range(self.n_eval_runs):
            eval_seed = self.seed + episode + 1
            obs, _ = env.reset(seed=eval_seed)

            done = False
            n_steps = 0
            speed_sum = 0.0
            episode_reward = 0.0
            last_info = {}

            while not done:
                action = self.act_numpy(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated
                n_steps += 1

                episode_reward += reward
                speed_sum += float(info.get("speedX", 0.0))
                last_info = info

            rewards[episode] = episode_reward
            mean_speed[episode] = speed_sum / max(n_steps, 1)

            distance[episode] = float(last_info.get("distRaced", 0.0))
            time_alive[episode] = float(last_info.get("timeAlive", 0.0))
            successful[episode] = int(last_info.get("successfulLap", False))
            damage[episode] = float(last_info.get("damage", 0.0))

        self.eval_stats.update(
            step=step,
            rewards=rewards,
            distance=distance,
            time_alive=time_alive,
            mean_speed=mean_speed,
            successful=successful,
            damage=damage,
        )

    def set_seeds(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.torch_rng.manual_seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def print_stats(self, step: int, episode: int, time: float) -> None:
        report = (
            f"Timestep: {step:>9d}  "
            f"Episode: {episode:>6d}  "
            f"Average Return: {self.eval_stats.last_average_reward:>12.4f}  "
            f"Std Return: {self.eval_stats.last_std_reward:>12.4f}  "
            f"Average Distance: {self.eval_stats.last_average_distance:>12.4f}  "
            f"Average Speed: {self.eval_stats.last_average_speed:>12.4f}  "
            f"Average Damage: {self.eval_stats.last_average_damage:>12.4f}  "
            f"Successful Laps: {self.eval_stats.last_successful_laps:>2d}/{self.n_eval_runs}  "
            f"Time: {time:>12.2f}s  "
        )
        print(report)


class OffPolicyAlgorithm(RLAlgorithm, ABC):
    def __init__(
        self,
        train_env: gym.Env,
        val_env: gym.Env, 
        *,
        gamma: float,
        tau_polyak: float,
        buffer_size: int,
        buffer_start_size: int,
        batch_size: int,
        gradient_steps: int,
        save_every: int,
        eval_every: int,
        n_eval_runs: int,
        verbose: bool = True,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            train_env=train_env,
            val_env=val_env, 
            gamma=gamma,
            tau_polyak=tau_polyak,
            save_every=save_every,
            eval_every=eval_every,
            n_eval_runs=n_eval_runs,
            verbose=verbose,
            seed=seed,
            device=device,
        )

        self.buffer_size = buffer_size
        self.buffer_start_size = buffer_start_size
        self.batch_size = batch_size
        self.gradient_steps = gradient_steps

        self.replay_buffer = ReplayBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            capacity=buffer_size,
            batch_size=batch_size,
            device=device,
        )

    def collect_rollouts(self) -> np.ndarray:
        env = self.env

        # This is ugly, but it works great... 
        if hasattr(env, "switch_every_steps") and hasattr(env, "tracks"):
            switch_every_steps = env.switch_every_steps
            n_tracks = len(env.tracks)
            env.switch_every_steps = int(self.buffer_start_size // n_tracks)
       
        env.action_space.seed(self.seed)
        obs, _ = env.reset(seed=self.seed)
        for _ in range(self.buffer_start_size): 
            action = env.action_space.sample()
            obs_next, reward, terminated, truncated, _ = env.step(action)
            self.replay_buffer.push(obs, action, reward, obs_next, terminated)
            obs = obs_next
            if terminated or truncated:
                obs, _ = env.reset()
        
        if hasattr(env, "switch_every_steps") and hasattr(env, "tracks"):
            env.switch_every_steps = switch_every_steps

        return obs

class OnPolicyAlgorithm(RLAlgorithm, ABC):
    def __init__(
        self,
        train_env: gym.Env,
        val_env: gym.Env, 
        *,
        gamma: float,
        tau_polyak: float,
        horizon: int,
        batch_size: int,
        epochs: int,
        save_every: int,
        eval_every: int,
        n_eval_runs: int,
        verbose: bool = True,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            train_env=train_env,
            val_env=val_env,  
            gamma=gamma,
            tau_polyak=tau_polyak,
            save_every=save_every,
            eval_every=eval_every,
            n_eval_runs=n_eval_runs,
            verbose=verbose,
            seed=seed,
            device=device,
        )

        self.horizon = horizon
        self.batch_size = batch_size
        self.epochs = epochs

        self.rollout_buffer = RolloutBuffer(
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            horizon=horizon,
            batch_size=batch_size,
        )