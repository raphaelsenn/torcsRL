from typing import List
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EvluationStats:
    steps_: List[int] = field(default_factory=list)
    average_rewards_: List[float] = field(default_factory=list)
    std_rewards_: List[float] = field(default_factory=list)

    def update(self, step: int, rewards: np.ndarray) -> None:
        average_return = float(np.mean(rewards).item())
        std_return = float(np.std(rewards).item())

        self.steps_.append(step)
        self.average_rewards_.append(average_return)
        self.std_rewards_.append(std_return)

    def to_csv(self, path: str) -> None:
        data = {
            "timesteps": self.steps_, 
            "average_return": self.average_rewards_, 
        }
        pd.DataFrame.from_dict(data).to_csv(path, index=False)

    @property
    def last_average_reward(self) -> float:
        return self.average_rewards_[-1]
    
    @property
    def last_std_reward(self) -> float:
        return self.std_rewards_[-1]