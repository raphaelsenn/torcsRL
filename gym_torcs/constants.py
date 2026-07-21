from __future__ import annotations

DEFAULT_PORT = 3001
SCR_MIN_PORT = 3001
SCR_MAX_PORT = 3010

DEFAULT_TORCS_EXECUTABLE = "/usr/local/bin/torcs"
DEFAULT_TORCS_LIB_DIRS = (
    "/usr/local/lib/torcs",
    "/usr/local/lib",
)

MAX_SPEED = 250.0                   # in [km/h]
MAX_ACCEL = 50.0                    # in [m/s^2]
MAX_TRACK = 200.0                   # in [m]
MAX_RPM = 10000.0                   # in [rounds per minute]
MAX_WHEEL_SPIN_VEL = 100.0
CURVATURE = 0.1

TERMINAL_JUDGE_START = 100          # If after 100 timesteps still no progress => terminate! (in timesteps [t])
TERMINATION_LIMIT_PROGRESS = 5.0    # Episode terminates if agent is runnig slower than that (in [km/h])

# Agressive
# Read more here: https://github.com/ugo-nama-kun/gym_torcs/blob/master/snakeoil3_gym.py
TRACK_SENSOR_ANGLES = (
    -45.0, -19.0, -12.0, -7.0, -4.0, -2.5, -1.7, -1.0, -0.5,
    0.0,
    0.5, 1.0, 1.7, 2.5, 4.0, 7.0, 12.0, 19.0, 45.0,
)

# Conservative
# Read more here: https://github.com/ugo-nama-kun/gym_torcs/blob/master/snakeoil3_gym.py
#TRACK_SENSOR_ANGLES = (
#    -90.0, -75.0, -60.0, -45.0, -30.0, -20.0, -15.0, -10.0, -5.0,
#    0.0,
#    5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0,
#)

# RACING_LINES = {}

RACING_LINES = {
    "alpine-1": "./data/alpine-1.csv",
    "alpine-2": "./data/alpine-2.csv",
    "street-1": "./data/street-1.csv",
    "brondehach": "./data/brondehach.csv",
    "ruudskogen" : "./data/ruudskogen.csv",
    "aalborg": "./data/aalborg.csv",
    "wheel-1": "./data/wheel-1.csv",
    "wheel-2": "./data/wheel-2.csv",
    "forza" : "./data/forza.csv",
    "g-track-1": "./data/g-track-1.csv",
    "g-track-2": "./data/g-track-2.csv",
    "g-track-3": "./data/g-track-3.csv",
    "corkscrew": "./data/corkscrew.csv",
}
