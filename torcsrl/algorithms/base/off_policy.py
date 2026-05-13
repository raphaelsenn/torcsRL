from abc import ABC
import gymnasium as gym

from torcsrl.config import EnvConfig
from torcsrl.algorithms.base.base import RLAlgorithm
from torcsrl.buffers.replay_buffer import ReplayBuffer


class OffPolicyAlgorithm(RLAlgorithm, ABC):
    def __init__(
        self,
        env_cfg: EnvConfig,
        *,
        gamma: float,
        tau_polyak: float,
        buffer_size: int,
        buffer_start_size: int,
        batch_size: int,
        gradient_steps: int,
        save_every: int,
        eval_every: int,
        n_eval_runs: int = 10,
        verbose: bool = True,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            env_cfg=env_cfg,
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

    def collect_rollouts(self) -> None:
        env = gym.make(self.env_id)
        obs, _ = env.reset(seed=self.seed)
        env.action_space.seed(self.seed)
        done = False
        for _ in range(self.buffer_start_size): 
            action = env.action_space.sample()
            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated 
            done_td = terminated
            self.replay_buffer.push(obs, action, reward, obs_next, done_td)
            obs = obs_next

            if done:
                obs, _ = env.reset()
                done = False