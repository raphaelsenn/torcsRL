from argparse import Namespace, ArgumentParser

import torch
import numpy as np

from torcsrl.algorithms.base import RLAlgorithm
from torcsrl.config import EnvConfig
from torcsrl import DDPG, TD3
from configs import DDPG_CONFIG, TD3_CONFIG


def algorithm_factory(args: Namespace) -> RLAlgorithm:
    env_cfg = EnvConfig() 
    if args.algorithm == "ddpg":
        return DDPG(
            env_cfg=env_cfg, 
            seed=args.seed, 
            device=args.device, 
            verbose=args.verbose, 
            **DDPG_CONFIG
        )
    elif args.algorithm == "td3":
        return TD3(
            env_cfg=env_cfg, 
            seed=args.seed, 
            device=args.device, 
            verbose=args.verbose, 
            **TD3_CONFIG
        )
    else:
        raise ValueError("Unkown algorithm.")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Torcs 1.3.7 training")

    parser.add_argument("--algorithm", type=str, default="DDPG")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", default=True)

    return parser.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    alg = algorithm_factory(args)
    alg.train(args.n_timesteps)


if __name__ == "__main__":
    main()