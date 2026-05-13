# torcsRL

## Install

1. Install [gym_torcs](https://github.com/raphaelsenn/gym_torcs) by following the instructions.

2. Activate the virtual environment from gym_torcs:

```bash
conda activate gym_torcs
```

3. Install using pip:

```bash
pip install -e .
```

## Usage

```python
import gymnasium as gym
import gym_torcs

from torcsrl import EnvConfig, DDPG


cfg = EnvConfig(
    executable = "/usr/local/bin/torcs"
    
    port_train = 3001 
    port_val = 3002

    track_category = "road"
    track_train = "forza"
    track_val = "ruudskogen"
)

alg = DDPG(
    env_cfg,
    lr_actor = 1e-3,
    lr_critic = 1e-3,
    gamma = 0.99,
)

alg.train(n_timesteps = 1_000_000)
```

## Citations

```bibtex
@misc{lillicrap2019continuouscontroldeepreinforcement,
      title={Continuous control with deep reinforcement learning}, 
      author={Timothy P. Lillicrap and Jonathan J. Hunt and Alexander Pritzel and Nicolas Heess and Tom Erez and Yuval Tassa and David Silver and Daan Wierstra},
      year={2019},
      eprint={1509.02971},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1509.02971}, 
}

@misc{fujimoto2018addressingfunctionapproximationerror,
      title={Addressing Function Approximation Error in Actor-Critic Methods}, 
      author={Scott Fujimoto and Herke van Hoof and David Meger},
      year={2018},
      eprint={1802.09477},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/1802.09477}, 
}
```
