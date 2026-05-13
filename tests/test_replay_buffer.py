from typing import Tuple

import numpy as np
import torch
import pytest

from torcsrl.buffers.replay_buffer import ReplayBuffer


OBS_DIM = 784
ACTION_DIM = 10
CAPACITY = 10
BATCH_SIZE = 8
DEVICE = "cpu"


@pytest.fixture
def replay_buffer() -> ReplayBuffer:
    return ReplayBuffer(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        capacity=CAPACITY,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )


@pytest.fixture
def transition() -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
    return (
        np.random.randn(OBS_DIM).astype(np.float32),
        np.random.randn(ACTION_DIM).astype(np.float32),
        float(np.random.randn()),
        np.random.randn(OBS_DIM).astype(np.float32),
        bool(np.random.randint(0, 2)),
    )


class TestReplayBuffer:
    def test_init(self, replay_buffer: ReplayBuffer) -> None:
        assert replay_buffer.obs_dim == OBS_DIM
        assert replay_buffer.action_dim == ACTION_DIM
        assert replay_buffer.capacity == CAPACITY
        assert replay_buffer.batch_size == BATCH_SIZE
        assert replay_buffer.position == 0
        assert replay_buffer.size == 0

    def test_push(
        self,
        replay_buffer: ReplayBuffer,
        transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool],
    ) -> None:
        for i in range(CAPACITY):
            replay_buffer.push(*transition)

            assert replay_buffer.size == i + 1
            assert replay_buffer.position == (i + 1) % replay_buffer.capacity

        for i in range(CAPACITY):
            replay_buffer.push(*transition)

            assert replay_buffer.size == replay_buffer.capacity
            assert replay_buffer.position == (i + 1) % replay_buffer.capacity

    def test_sample(
        self,
        replay_buffer: ReplayBuffer,
        transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool],
    ) -> None:
        for _ in range(CAPACITY):
            replay_buffer.push(*transition)

        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = replay_buffer.sample()

        assert isinstance(obs_batch, torch.Tensor)
        assert isinstance(action_batch, torch.Tensor)
        assert isinstance(reward_batch, torch.Tensor)
        assert isinstance(next_obs_batch, torch.Tensor)
        assert isinstance(done_batch, torch.Tensor)

        assert obs_batch.dtype == torch.float32
        assert action_batch.dtype == torch.float32
        assert reward_batch.dtype == torch.float32
        assert next_obs_batch.dtype == torch.float32
        assert done_batch.dtype == torch.float32

        assert obs_batch.shape == (BATCH_SIZE, OBS_DIM)
        assert action_batch.shape == (BATCH_SIZE, ACTION_DIM)
        assert reward_batch.shape == (BATCH_SIZE,)
        assert next_obs_batch.shape == (BATCH_SIZE, OBS_DIM)
        assert done_batch.shape == (BATCH_SIZE,)

        assert obs_batch.device.type == DEVICE
        assert action_batch.device.type == DEVICE
        assert reward_batch.device.type == DEVICE
        assert next_obs_batch.device.type == DEVICE
        assert done_batch.device.type == DEVICE