import gymnasium as gym
from gym_torcs import TorcsEnv

import torch
import numpy as np
# from torcsrl.models.ac_td3 import ActorMLP
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
    # racing_line_csv="alpine_2_tita_expert.csv",
    laps=20,
    debug=True,
    gui_auto_start=True,
    observation_noise=True
)

actor = ActorMLP(np.sum(env.observation_space.shape), 2)
# actor.load_state_dict(torch.load("actor-3245000.pt", map_location="cpu"))
actor.load_state_dict(torch.load("actor-4695000-seed2.pt", map_location="cpu"))

seed = 0
# np.random.seed(seed)
for ep in range(2):
    obs, info = env.reset()
    # obs, info = env.reset(seed=seed + ep)
    done = False
    t = 0
    total_reward = 0
    print(info)
    print(obs)
    while not done:
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0) 
        action = actor.act(obs).cpu().detach().numpy().flatten()
        #if obs[0, 0] > 0.4: 
        #    action[0] = action[0].clip(-0.1, 0.1) 
        #elif 0.2 < obs[0, 0] <= 0.4:
        #    action[0] = action[0].clip(-0.5, 0.5) 
        #else:
        #    action[0] = action[0]
        # action = env.action_space.sample()
        obs_next, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        # print(total_reward, info, terminated, truncated) 
        print(f"T: {t}\tTotal Return: {total_reward:.2f}\tDistance Raced: {info['distRaced']:.2f}\tReward: {reward:.2f}")
        obs = obs_next
        t += 1

    print(f"Episode {ep + 1} with reward: {reward}")

env.close()