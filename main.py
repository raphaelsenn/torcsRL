import gymnasium as gym
import gym_torcs

from torcsrl.config import EnvConfig
from torcsrl import DDPG, TD3


cfg = EnvConfig()

alg = TD3(cfg, lr_actor=0.0003, lr_critic=0.0003, buffer_start_size=25_000, batch_size=64, tau_polyak=0.005, save_every=5_000, eval_every=5_000)
# alg = DDPG(cfg, lr_actor=0.0001, lr_critic=0.001, buffer_start_size=25_000, batch_size=64, tau_polyak=0.001, save_every=5_000, eval_every=5_000)
alg.train(500_000)