import gymnasium as gym
import gym_torcs

from torcsrl.config import EnvConfig
from torcsrl import DDPG, TD3


cfg = EnvConfig()



# alg = TD3(cfg, lr_actor=0.0003, lr_critic=0.0003, buffer_start_size=25_000, batch_size=256, tau_polyak=0.005, save_every=5_000, eval_every=5_000)
#seeds = [0, 1, 2]
#for seed in seeds:
#    alg = DDPG(cfg, lr_actor=0.0003, lr_critic=0.0003, buffer_size=1_000_000, buffer_start_size=25_000, batch_size=256, tau_polyak=0.005, save_every=5_000, eval_every=5_000, seed=seed)
#    alg.train(3_000_000)
alg = DDPG(cfg, lr_actor=0.0003, lr_critic=0.0003, buffer_size=1_000_000, buffer_start_size=25_000, batch_size=256, tau_polyak=0.005, save_every=5_000, eval_every=5_000, seed=3)
alg.train(1_000_000)