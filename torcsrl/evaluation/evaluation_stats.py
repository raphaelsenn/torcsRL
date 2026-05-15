from typing import List
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EvluationStats:
    steps_: List[int] = field(default_factory=list)
    average_rewards_: List[float] = field(default_factory=list)
    std_rewards_: List[float] = field(default_factory=list)
    
    average_speed_: List[float] = field(default_factory=list)
    
    average_distance_: List[float] = field(default_factory=list)
    min_distance_: List[float] = field(default_factory=list)
    max_distance_: List[float] = field(default_factory=list)

    average_time_alive_: List[float] = field(default_factory=list)
    min_time_alive_: List[float] = field(default_factory=list)
    max_time_alive_: List[float] = field(default_factory=list)

    def update(
            self, 
            step: int, 
            rewards: np.ndarray, 
            distance: np.ndarray, 
            time_alive: np.ndarray, 
            mean_speed: np.ndarray, 
    ) -> None:
        average_return = float(np.mean(rewards).item())
        std_return = float(np.std(rewards).item())
        
        average_speed = float(np.mean(mean_speed).item())
        
        average_distance = float(np.mean(distance).item())
        min_distance = float(np.min(distance))
        max_distance = float(np.max(distance))

        average_time_alive = float(np.mean(time_alive).item())
        min_time_alive = float(np.min(time_alive))
        max_time_alive = float(np.max(time_alive))

        self.steps_.append(step)
        self.average_rewards_.append(average_return)
        self.std_rewards_.append(std_return)

        self.average_speed_.append(average_speed)

        self.average_distance_.append(average_distance)
        self.min_distance_.append(min_distance)
        self.max_distance_.append(max_distance)

        self.average_time_alive_.append(average_time_alive)
        self.min_time_alive_.append(min_time_alive)
        self.max_time_alive_.append(max_time_alive)

    def to_csv(self, path: str) -> None:
        data = {
            "timesteps": self.steps_, 
            
            "average_return": self.average_rewards_,
            "std_return": self.std_rewards_,
            
            "average_speed": self.average_speed_,
            
            "average_distance": self.average_distance_,
            "min_distance": self.min_distance_,
            "max_distance": self.max_distance_,
            
            "average_time_alive": self.average_time_alive_,
            "min_time_alive": self.min_time_alive_,
            "max_time_alive": self.max_time_alive_,
        }
        pd.DataFrame.from_dict(data).to_csv(path, index=False)

    @property
    def last_average_reward(self) -> float:
        return self.average_rewards_[-1]
    
    @property
    def last_std_reward(self) -> float:
        return self.std_rewards_[-1]
    
    @property
    def last_average_speed(self) -> float:
        return self.average_speed_[-1]
    
    @property
    def last_average_distance(self) -> float:
        return self.average_distance_[-1]
    
    @property
    def last_min_distance(self) -> float:
        return self.min_distance_[-1]
    
    @property
    def last_max_distance(self) -> float:
        return self.max_distance_[-1]
    
    @property
    def last_average_time_alive(self) -> float:
        return self.average_time_alive_[-1]
    
    @property
    def last_min_time_alive(self) -> float:
        return self.min_time_alive_[-1]
    
    @property
    def last_max_time_alive(self) -> float:
        return self.max_time_alive_[-1]