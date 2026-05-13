from dataclasses import dataclass


@dataclass
class EnvConfig:
    executable: str = "/usr/local/bin/torcs"
    
    port_train: int = 3001 
    port_val: int = 3002

    track_category: str = "road"
    track_train: str = "forza"
    track_val: str = "ruudskogen"