import gymnasium as gym
from gym_torcs import TorcsEnv

import torch
import numpy as np
from torcsrl.models.ac_ddpg import ActorMLP


# Launch TORCS automatically
#env = TorcsEnv(render_mode="human", throttle=True, max_episode_steps=10000, port=3001)
env = gym.make(
    "TorcsSCR-v0",
    render_mode="human",        # or None
    executable="/usr/local/bin/torcs",
    port=3003,                  # 3001..3010
    track_name="wheel-2",
    track_category="road",
    laps=20,
    debug=True,
    gui_auto_start=True,
)

actor = ActorMLP(np.sum(env.observation_space.shape), 2)
actor.load_state_dict(torch.load("actor-410000.pt", map_location="cpu"))

seed = 0
np.random.seed(seed)
for ep in range(2):
    obs, info = env.reset(seed=seed + ep)
    done = False
    t = 0
    total_reward = 0
    print(info) 
    while not done:
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0) 
        action = actor.act(obs).cpu().detach().numpy().flatten()
        # action = env.action_space.sample()
        obs_next, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        # print(total_reward, info, terminated, truncated) 
        print(f"Total Return: {total_reward}\tDistance Raced: {info['distRaced']}")
        obs = obs_next

    print(f"Episode {ep + 1} with reward: {reward}")

env.close()