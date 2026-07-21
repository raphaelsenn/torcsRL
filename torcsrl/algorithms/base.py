from abc import ABC, abstractmethod
from typing import Dict, Any

import os
import time
import random
import torch
import numpy as np
import gymnasium as gym

from torcsrl.models.base import Actor, Critic
from torcsrl.evaluation.evaluation_stats import EvluationStats
from torcsrl.buffers.replay_buffer import ReplayBuffer
from torcsrl.buffers.rollout_buffer import RolloutBuffer


class RLAlgorithm(ABC):
    """Base actor-critic algorithm class.""" 
    def __init__(
        self,
        train_env: gym.Env,
        val_env: gym.Env,
        *,
        ac_kwargs: Dict[str, Any],
        gamma: float,
        tau_polyak: float,
        save_every: int, 
        eval_every: int, 
        verbose: bool = True,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        ...
        super().__init__()

        self.env = train_env
        self.val_env = val_env

        # Computing device
        assert device in {"cpu", "cuda", "mps"}, (
            f"Unkdown device, expected device in [`cpu`, `cuda`, `mps`], got: {device}."
        )
        self.device = torch.device(device)

        # Actor/critic models
        self.ac_kwargs = ac_kwargs
        self.actor: Actor = ...
        self.critic: Critic = ...

        # Environment
        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space
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

        # Hyperparameters and save/eval settings
        self.gamma = gamma
        self.tau_polyak = tau_polyak
        self.save_every = save_every
        self.eval_every = eval_every 
        self.verbose = verbose
        
        # Counters
        self.episode_counter = 0
        self.grad_step_counter = 0
        self.global_step_counter = 0

        # Evaluation
        self.eval_stats: dict[str, EvluationStats] = {}
        if hasattr(self.val_env, "tracks"):
            for track in self.val_env.tracks:
                key = f"{track.category}/{track.name}"
                self.eval_stats[key] = EvluationStats()
        else:
            self.eval_stats["eval"] = EvluationStats()

        # Reproducability
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
    def collect_rollouts(self) -> tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError

    @torch.no_grad()
    def act_numpy(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)   # [obs_dim]
        obs_t = obs_t.unsqueeze(0)                                              # [1, obs_dim]
        action = self.act(obs_t, deterministic)                                 # [1, action_dim]
        action = action.flatten().cpu().numpy()                                 # [action_dim]
        return action

    @torch.no_grad()
    def evaluate(self) -> None:
        env = self.val_env

        if hasattr(env, "tracks"):
            n_eval_tracks = len(env.tracks)
        else:
            n_eval_tracks = 1

        for episode in range(n_eval_tracks):
            eval_seed = self.seed + episode + 1
            obs, reset_info = env.reset(seed=eval_seed)

            track = reset_info.get("track", None)
            if track is not None:
                track_name = str(track["name"])
                track_category = str(track["category"])
                track_key = f"{track_category}/{track_name}"
            else:
                track_key = "eval"

            if track_key not in self.eval_stats:
                self.eval_stats[track_key] = EvluationStats()

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

            rewards = np.asarray([episode_reward], dtype=np.float32)
            distance = np.asarray([float(last_info.get("distRaced", 0.0))], dtype=np.float32)
            time_alive = np.asarray([float(last_info.get("timeAlive", 0.0))], dtype=np.float32)
            mean_speed = np.asarray([speed_sum / max(n_steps, 1)], dtype=np.float32)
            successful = np.asarray([int(last_info.get("successfulLap", False))], dtype=np.uint8)
            damage = np.asarray([float(last_info.get("damage", 0.0))], dtype=np.float32)
            lap_time = np.asarray(float(last_info.get("lastLapTime", 0.0)), dtype=np.float32)

            self.eval_stats[track_key].update(
                step=self.global_step_counter,
                rewards=rewards,
                distance=distance,
                time_alive=time_alive,
                mean_speed=mean_speed,
                successful=successful,
                damage=damage,
                lap_time=lap_time
            )

    def set_seeds(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.torch_rng.manual_seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def on_episode_end(
        self,
        pending_eval: bool,
        pending_save: bool,
        start_time: float,
    ) -> tuple[bool, bool, float]:
        if pending_eval:
            self.evaluate()

            elapsed = time.monotonic() - start_time
            start_time = time.monotonic()

            if self.verbose:
                self.print_stats(elapsed)

            # Save exact model that produced printed eval stats
            self.save()

            pending_eval = False
            pending_save = False

        elif pending_save:
            self.save()
            pending_save = False

        return pending_eval, pending_save, start_time

    def print_stats(self, time: float) -> None:
        header = (
            f"Timestep: {self.global_step_counter:>9d}  "
            f"Episode: {self.episode_counter:>6d}  "
            f"Time: {time:>12.2f}s"
        )
        print(header)

        for track_key, stats in self.eval_stats.items():
            if len(stats.steps_) == 0:
                continue

            report = (
                f"  Track: {track_key:<20s}  "
                f"Return: {stats.last_average_reward:>12.4f}  "
                f"Distance: {stats.last_average_distance:>12.4f}  "
                f"Speed: {stats.last_average_speed:>10.4f}  "
                f"Damage: {stats.last_average_damage:>10.4f}  "
                f"Lap time: {stats.last_lap_time:>12.4f}  "
                f"Lap: {stats.last_successful_laps:>1d}/1"
            )
            print(report)
    
    def save(self) -> None:
        algo_name = self.__class__.__name__.lower()
        run_name = f"{algo_name}-lr_pi{self.lr_actor}-lr_q{self.lr_critic}-seed{self.seed}"
        save_dir = os.path.join("checkpoints", algo_name, run_name)
        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.actor.state_dict(), os.path.join(save_dir, f"actor-{algo_name}-{self.global_step_counter}.pt"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, f"critic-{algo_name}-{self.global_step_counter}.pt"))
        
        for track_key, stats in self.eval_stats.items():
            safe_track_key = track_key.replace("/", "_")
            save_dir = os.path.join("checkpoints", algo_name, run_name, f"eval_stats_{safe_track_key}.csv")
            stats.to_csv(save_dir)



class OffPolicyAlgorithm(RLAlgorithm, ABC):
    def __init__(
        self,
        train_env: gym.Env,
        val_env: gym.Env, 
        *,
        ac_kwargs: Dict[str, Any],
        gamma: float,
        tau_polyak: float,
        buffer_size: int,
        buffer_start_size: int,
        batch_size: int,
        gradient_steps: int,
        save_every: int,
        eval_every: int,
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
            save_every=save_every,
            eval_every=eval_every,
            verbose=verbose,
            seed=seed,
            device=device,
        )

        self.buffer_size = buffer_size
        self.buffer_start_size = buffer_start_size
        self.batch_size = batch_size
        self.gradient_steps = gradient_steps

        self.replay_buffer = ReplayBuffer(
        # self.replay_buffer = ReplayBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            capacity=buffer_size,
            batch_size=batch_size,
            device=device,
        )

    def collect_rollouts(self) -> tuple[np.ndarray, Dict[str, Any]]:
        env = self.env  # NOTE: Training environment

        env.action_space.seed(self.seed)
        obs, info = env.reset(seed=self.seed)
        
        for _ in range(self.buffer_start_size):
            # Sample random action and step environment.
            action = env.action_space.sample()
            obs_next, reward, terminated, truncated, _ = env.step(action)
            
            # Save in replay memory.
            self.replay_buffer.push(obs, action, reward, obs_next, terminated)

            obs = obs_next
            if terminated or truncated:
                obs, info = env.reset()

        return obs, info

    def collect_expert_rollouts(self) -> tuple[np.ndarray, Dict[str, Any]]:
        env = self.env  # NOTE: Training environment

        env.action_space.seed(self.seed)
        obs, info = env.reset(seed=self.seed)

        for _ in range(self.buffer_start_size):
            # Use expert tita action 
            action = np.asarray(info["action_tita"], dtype=np.float32)
            obs_next, reward, terminated, truncated, info = env.step(action)
            
            # Save in replay memory.
            self.replay_buffer.push(obs, action, reward, obs_next, terminated)

            obs = obs_next
            if terminated or truncated:
                print(f"|B| = {len(self.replay_buffer)}")
                obs, info = env.reset()

        return obs, info


class OnPolicyAlgorithm(RLAlgorithm, ABC):
    def __init__(
        self,
        train_env: gym.Env,
        val_env: gym.Env, 
        *,
        ac_kwargs: Dict[str, Any],
        gamma: float,
        tau_polyak: float,
        horizon: int,
        batch_size: int,
        epochs: int,
        save_every: int,
        eval_every: int,
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
            save_every=save_every,
            eval_every=eval_every,
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
