from typing import Tuple
import numpy as np

from torcsrl.buffers.base import BufferBase


class RolloutBuffer(BufferBase):
    def __init__(
            self, 
            obs_dim: Tuple, 
            action_dim: Tuple, 
            horizon: int, 
            batch_size: int,
    ) -> None:
        super().__init__(obs_dim, action_dim, horizon)
        self.batch_size = batch_size
        self.advantages = np.empty(shape=(horizon,), dtype=np.float32)