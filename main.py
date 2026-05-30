import gymnasium as gym
import gym_torcs

from torcsrl import DDPG, TD3
from gym_torcs.wrappers import TrackSpec, TimedTrackSelectionWrapper

train_tracks = [
    TrackSpec("alpine-1", "road", "alpine_1_tita_expert.csv"),
    TrackSpec("g-track-1", "road", "g_track_1_tita_expert.csv"),
    TrackSpec("street-1", "road", "street_1_tita_expert.csv"),
    TrackSpec("brondehach", "road", "brondehach_tita_expert.csv"),
    TrackSpec("aalborg", "road", "aalborg_tita_expert.csv"),
    TrackSpec("e-track-2", "road", "e_track_2_tita_expert.csv"),
    TrackSpec("forza", "road", "forza_tita_expert.csv"),
]


seeds = [0, 2]
for seed in seeds:
    train_env = gym.make(
        "TorcsSCR-v0",
        render_mode=None,
        port=3001,
        race_type="practice",
        track_name="alpine-1",
        track_category="road",
        reset_strategy="relaunch",
        racing_line_csv="alpine_1_tita_expert.csv",
    )

    train_env = TimedTrackSelectionWrapper(
        train_env, train_tracks, switch_every_steps=50_000
    )

    val_env = gym.make(
        "TorcsSCR-v0",
        render_mode=None,
        port=3002,
        race_type="practice",
        track_name="alpine-2",
        track_category="road",
        reset_strategy="relaunch",
        racing_line_csv="alpine_2_tita_expert.csv",
    )

    # alg = DDPG(train_env, val_env, gamma=0.99, lr_actor=0.0001, lr_critic=0.001, buffer_size=5_000_000, buffer_start_size=50_000, batch_size=32, tau_polyak=0.001, save_every=5_000, eval_every=5_000, device="cuda", seed=seed)
    alg = TD3(train_env, val_env, lr_actor=0.0001, lr_critic=0.001, buffer_size=5_000_000, buffer_start_size=50_000, batch_size=32, tau_polyak=0.001, save_every=5_000, eval_every=5_000, device="cuda", seed=seed)
    print(f"Start training {alg.__class__.__name__} on seed = {seed}") 
    alg.train(5_000_000)

#seeds = [0, 1, 2]
#for seed in seeds:
#    print(f"Start training DDPG on seed = {seed}") 
#    alg = DDPG(train_env, val_env, lr_actor=0.0003, lr_critic=0.0003, buffer_size=1_000_000, buffer_start_size=25_000, batch_size=256, tau_polyak=0.005, save_every=5_000, eval_every=5_000, device="cuda", seed=seed)
#    alg.train(1_000_000)

# alg = TD3(cfg, lr_actor=0.0003, lr_critic=0.0003, buffer_start_size=25_000, batch_size=256, tau_polyak=0.005, save_every=5_000, eval_every=5_000)
#seeds = [0, 1, 2]
#for seed in seeds:
#    alg = TD3(cfg, lr_actor=0.0003, lr_critic=0.0003, buffer_size=1_000_000, buffer_start_size=25_000, batch_size=256, tau_polyak=0.005, save_every=5_000, eval_every=5_000, device="cuda", seed=seed)
#    alg.train(1_000_000)
# alg = DDPG(cfg, lr_actor=0.0003, lr_critic=0.0003, buffer_size=1_000_000, buffer_start_size=25_000, batch_size=256, tau_polyak=0.005, save_every=5_000, eval_every=5_000, seed=3)
# alg.train(1_000_000)