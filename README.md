# torcsRL

## Training and Evaluation

| Purpose | Track | Preview |
| --- | --- | --- |
| Training | `street-1` | ![TORCS street-1 training track](./assets/torcs_street-1.png) |
| Evaluation | `ruudskogen` | ![TORCS ruudskogen evaluation track](./assets/torcs_ruudskogen.png) |


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

## Hyperparameters

| Algorithm | Timesteps | Learning rate actor | Learning rate critic | Policy delay | Batch size | Replay buffer size | Start steps | Gamma | Polyak tau | Exploration noise |
| --------- | --------- | ------------------- | -------------------- | ------------ | ---------- | ------------------ | ----------- | ----- | ---------- | ----------------- |
| DDPG | 1,000,000 | 1e-4 | 1e-3 | – | 64 | 1,000,000 | 25,000 | 0.99 | 0.001 | Gaussian, $\sigma$ = 0.1 | 
| TD3 | 1,000,000 | 1e-4 | 1e-3 | 2 | 256 | 1,000,000 | 25,000 | 0.99 | 0.005 | Gaussian, $\sigma$ = 0.1 |


## Install

## Fedora Linux 44 Workstation

### Install TORCS 1.3.7 with the SCR server patch

1. Clone the TORCS repository:

```bash
git clone https://github.com/raphaelsenn/torcs-1.3.7
```

2. Enter the repository:

```bash
cd torcs-1.3.7
```

3. Install the required packages:

```bash
sudo dnf install \
  glib2-devel \
  mesa-libGL-devel \
  mesa-libGLU-devel \
  freeglut-devel \
  plib-devel \
  openal-soft-devel \
  freealut-devel \
  libXi-devel \
  libXmu-devel \
  libXrender-devel \
  libXrandr-devel \
  libpng-devel \
  libvorbis-devel \
  gcc \
  gcc-c++ \
  make \
  cmake \
  automake \
  autoconf \
  libtool \
  libXxf86vm-devel
```

4. Build and install TORCS:

```bash
make
sudo make install
sudo make datainstall
```

You should now be able to start TORCS with:

```bash
sudo torcs
```

### Install `torcsrl`

1. Create and activate a conda environment:

```bash
conda create -n torcsrl python=3.11 -y
conda activate torcsrl
```

2. Clone the repository:

```bash
git clone https://github.com/raphaelsenn/gym_torcs
cd gym_torcs
```

3. Install the Python requirements:

```bash
pip install -r requirements.txt
```

4. Install `torcsrl` in editable mode:

```bash
pip install -e .
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
