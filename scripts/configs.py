
DDPG_CONFIG = {
    "lr_actor": 0.0003,
    "lr_critic": 0.0003,
    "buffer_size": 1_000_000,
    "buffer_start_size": 25_000,
    "batch_size": 256,
    "epsilon" : 0.1,
    "tau_polyak": 0.005,
    "save_every": 5_000,
    "eval_every": 5_000,
}


TD3_CONFIG = {
    "lr_actor": 0.0003,
    "lr_critic": 0.0003,
    "buffer_size": 1_000_000,
    "buffer_start_size": 25_000,
    "batch_size": 256,
    "tau_polyak": 0.005,
    "save_every": 5_000,
    "eval_every": 5_000,
}