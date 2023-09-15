from enum import Enum

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStack


class TaskName(Enum):
    """To avoid using strings when checking for task, this enum is used."""

    BALANCE = "balance"
    REF_TRACKING = "reference_tracking"
    MOTION_CONTROL = "motion_control"
    GO_FORWARD = "go_forward"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            # print(self.value, other.value, type(self.value), self.value == other.value)
            return self.value == other.value
        elif isinstance(other, str):
            # print(other, self)
            return self.value == other
        else:
            return self.value == other.value
        return False


class ActionFrameStack(FrameStack):
    def __init__(
        self,
        env: gym.Env,
        num_stack: int,
        stack_action: bool,
        lz4_compress: bool = False,
    ):
        super().__init__(env, num_stack, lz4_compress)
        self.stack_action = stack_action
        if stack_action:
            low_single = np.concatenate(
                (self.env.observation_space.low, self.env.action_space.low)
            )
            high_single = np.concatenate(
                (self.env.observation_space.high, self.env.action_space.high)
            )
        else:
            low_single = self.env.observation_space.low
            high_single = self.env.observation_space.high
        low = np.repeat(low_single[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(high_single[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.stack_action:
            observation = np.concatenate((observation, action))
        self.frames.append(observation)
        return self.observation(None), reward, terminated, truncated, info

    def last_observation(self):
        return self.frames[-1][: np.prod(self.env.observation_space.shape)]

    def reset(self, **kwargs):
        """Reset the environment with kwargs.
        Args:
            **kwargs: The kwargs for the environment reset
        Returns:
            The stacked observations
        """
        obs, info = self.env.reset(**kwargs)
        if self.stack_action:
            obs = np.concatenate((obs, self.env.action_space.low * 0))
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self.observation(obs), info


def randomize_parameter(mean, param_noise_percent_std, affine: float = 0):
    if isinstance(mean, np.ndarray):
        shape = mean.shape
    else:
        shape = [1]
    ret = (
        np.random.randn(*shape) * (mean + affine) * param_noise_percent_std / 100 + mean
    )
    if ret.shape == (1,):
        return ret[0]
    else:
        return ret
