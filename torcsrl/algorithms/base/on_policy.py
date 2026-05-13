from abc import ABC
import gymnasium as gym

from torcsrl.config import EnvConfig
from torcsrl.algorithms.base.base import RLAlgorithm
from torcsrl.buffers.rollout_buffer import RolloutBuffer


class OnPolicyAlgorithm(RLAlgorithm, ABC):
    def __init__(
        self,
        env_cfg: EnvConfig,
        *,
        gamma: float,
        tau_polyak: float,
        horizon: int,
        batch_size: int,
        epochs: int,
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

        self.horizon = horizon
        self.batch_size = batch_size
        self.epochs = epochs

        self.rollout_buffer = RolloutBuffer(
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            horizon=horizon,
            batch_size=batch_size,
        )