import numpy as np


class BufferBase:
    def __init__(
            self, 
            obs_dim: int, 
            action_dim: int, 
            capacity: int,
    ) -> None:
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.position = 0
        self.size = 0

        self.obs = np.empty(shape=(capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty(shape=(capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty(shape=(capacity,), dtype=np.float32)
        self.obs_next = np.empty(shape=(capacity, obs_dim), dtype=np.float32)
        self.dones = np.empty(shape=(capacity,), dtype=np.float32)

    def push(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: float, 
            obs_next: np.ndarray, 
            dones: bool,
    ) -> None:
        pos = self.position

        self.obs[pos] = obs.astype(np.float32)
        self.actions[pos] = action.astype(np.float32)
        self.rewards[pos] = float(reward)
        self.obs_next[pos] = obs_next.astype(np.float32)
        self.dones[pos] = float(dones)

        self.position = (pos + 1) % self.capacity 
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return self.size