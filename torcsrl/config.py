from dataclasses import dataclass


@dataclass
class EnvConfig:
    executable: str = "/usr/local/bin/torcs"
    
    port_train: int = 3001 
    port_val: int = 3002

    track_category: str = "road"
    track_train: str = "alpine-1"       # 6355.65 meters
    track_val: str = "alpine-2"         # 3773.57 meters