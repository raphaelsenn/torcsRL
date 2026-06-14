import gymnasium as gym
import gym_torcs

from torcsrl import DDPG, TD3, DDPG_LAP, NSTEP_DDPG, NSTEP_TD3
from gym_torcs.wrappers import TrackSpec, TimedTrackSelectionWrapper, HistoryWrapper, ActionSmoothingWrapper


HORIZON_OBS = 3
HORIZON_ACT = 2

train_tracks = [
    TrackSpec("alpine-1", "road"),
    TrackSpec("aalborg", "road"),
    TrackSpec("brondehach", "road"),
    TrackSpec("e-track-2", "road"),
    TrackSpec("street-1", "road"),
    TrackSpec("forza", "road"),
]

val_tracks = [
    # Training tracks
    TrackSpec("alpine-1", "road"),
    TrackSpec("aalborg", "road"),
    TrackSpec("street-1", "road"),
    TrackSpec("brondehach", "road"),
    TrackSpec("e-track-2", "road"),
    TrackSpec("forza", "road"),
    # TrackSpec("g-track-1", "road", "g_track_1_tita_expert.csv"),

    # Validation tracks
    TrackSpec("ruudskogen", "road"),
    TrackSpec("wheel-2", "road"),
    TrackSpec("alpine-2", "road"),
]


seeds = [0, 2, 3]
for seed in seeds:
    train_env = gym.make(
        "TorcsSCR-v0",
        render_mode=None,
        port=3001,
        race_type="practice",
        track_name="alpine-1",
        track_category="road",
        reset_strategy="relaunch",
        truncate_on_successful_lap=False,
        observation_noise=False
    )

    # train_env = ActionSmoothingWrapper(train_env, 0.7)
    train_env = HistoryWrapper(train_env, HORIZON_OBS, HORIZON_ACT)
    train_env = TimedTrackSelectionWrapper(
        train_env, train_tracks, switch_every_steps=10_000
    )

    val_env = gym.make(
        "TorcsSCR-v0",
        render_mode=None,
        port=3002,
        race_type="practice",
        track_name="alpine-2",
        track_category="road",
        reset_strategy="relaunch",
        truncate_on_successful_lap=True,
        observation_noise=False
    )
    
    # val_env = ActionSmoothingWrapper(val_env, 0.7)
    val_env = HistoryWrapper(val_env, HORIZON_OBS, HORIZON_ACT)
    val_env = TimedTrackSelectionWrapper(
        val_env, val_tracks, switch_every_steps=1
    )
    
    ac_kwargs = {"h1_dim": 256, "h2_dim": 256}
    # alg = DDPG_LAP(train_env, val_env, ac_kwargs=ac_kwargs, lap_alpha=0.4, gamma=0.992, lr_actor=0.0003, lr_critic=0.0003, buffer_size=1_000_000, buffer_start_size=25_000, batch_size=256, tau_polyak=0.005, save_every=10_000, eval_every=10_000, device="cuda", seed=seed)
    # alg = NSTEP_DDPG(train_env, val_env, ac_kwargs=ac_kwargs, n_steps=5, gamma=0.992, lr_actor=0.0003, lr_critic=0.0003, buffer_size=1_000_000, buffer_start_size=100_000, batch_size=256, tau_polyak=0.005, save_every=10_000, eval_every=10_000, device="cuda", seed=seed)
    # alg = DDPG(train_env, val_env, ac_kwargs=ac_kwargs, gamma=0.99, lr_actor=0.0003, lr_critic=0.0003, buffer_size=1_000_000, buffer_start_size=100_000, batch_size=256, tau_polyak=0.005, save_every=10_000, eval_every=10_000, device="cuda", seed=seed)
    # alg = DDPG_BC("brr", train_env, ac_kwargs=ac_kwargs, gamma=0.99, lr_actor=0.0003, lr_critic=0.0003, buffer_size=1_000_000, buffer_start_size=100_000, batch_size=256, tau_polyak=0.005, save_every=10_000, eval_every=10_000, device="cuda", seed=seed)
    # alg = TD3(train_env, val_env, lr_actor=0.00001, lr_critic=0.0001, gamma=0.993, buffer_size=200_000, buffer_start_size=2_000, batch_size=64, tau_polyak=0.001, save_every=10_000, eval_every=10_000, device="cuda", seed=seed)
    alg = NSTEP_TD3(train_env, val_env, n_steps = 3, ac_kwargs=ac_kwargs, lr_actor=0.0003, lr_critic=0.0003, gamma=0.99, buffer_size=1_000_000, buffer_start_size=10_000, batch_size=256, tau_polyak=0.005, save_every=10_000, eval_every=10_000, device="cuda", seed=seed)
    print(f"Start training {alg.__class__.__name__} on seed = {seed}, params actor = {sum(p.numel() for p in alg.actor.parameters())}")
    print(alg.action_scale_np, alg.action_bias_np)
    alg.train(3_000_000)

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