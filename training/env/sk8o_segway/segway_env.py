from copy import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from omegaconf import OmegaConf
from sk8o_sim import SegwaySimData, SegwaySimulation, SegwaySimulationCfg

from ..common import TaskName
from .reward import Reward


class SegwayEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env_cfg: OmegaConf,
        task_cfg: OmegaConf,
        evaluation: bool = False,
        render_mode: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        env_cfg : OmegaConf
            Environment configuration from the training configuration.
        task_cfg : OmegaConf
            Task configuration.
        evaluation : bool, optional
            Whether this is an evaluation environment (as opposed to training), by default False
        """

        self.action_names = ["wheel_L", "wheel_R"]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.sim = SegwaySimulation(
            SegwaySimulationCfg(**env_cfg.simulation),
        )
        self.render_mode = render_mode

        self.reward = Reward(task_cfg, evaluation)
        self.change_observability(task_cfg.observability)
        self.change_control_frequency(task_cfg.control_frequency)
        self.task_name = TaskName(task_cfg.name)
        if evaluation:
            task_cfg = task_cfg.eval
        else:
            task_cfg = task_cfg.train
        if self.task_name in (
            TaskName.MOTION_CONTROL,
            TaskName.GO_FORWARD,
            TaskName.BALANCE,
        ):
            self.velocity_reference_low = (
                task_cfg.reference.forward_velocity_range[0],
                task_cfg.reference.angular_velocity_range[0],
            )
            self.velocity_reference_high = (
                task_cfg.reference.forward_velocity_range[1],
                task_cfg.reference.angular_velocity_range[1],
            )
        elif self.task_name in (TaskName.REF_TRACKING,):
            self.position_reference_low = task_cfg.reference.position.low
            self.position_reference_high = task_cfg.reference.position.high

    def change_observability(self, observability: List[str]):
        """Changes the variables included in the gym observation.

        Parameters
        ----------
        observability : List[str]
            A list of names of variables to be included.
        """
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(observability),), dtype=np.float64
        )
        self.observation_names = observability
        SegwaySimData.to_gym = self.gym_masker(observability)

    def change_control_frequency(self, control_frequency: float):
        self.control_frequency = control_frequency

    def render(self) -> Optional[np.ndarray]:
        """Render the environement (AI Gym interface method).

        Parameters
        ----------
        mode : str, optional
            What render mode to use (see the metadata), by default "human"

        Returns
        -------
        Optional[np.ndarray]
            The rendered data (or None, if rendered to screen).
        """
        return self.sim.render()

    def segway_obs(self, data: Optional[SegwaySimData]) -> np.ndarray:
        # for compatibility with sk8o_full
        if data is None:
            data = self.data
        return data.to_gym()

    def new_reference(
        self, reference: Tuple[float, float] | Tuple[float, float, float] | None
    ):
        """Sets a position or velocity reference based on the argument or configuration.

        The env always assumes only one reference.

        Parameters
        ----------
        reference : Tuple[float, float] | Tuple[float, float, float] | None
            The reference to be set. If 2-tuple, velocity reference will be set. If 3-tuple, position reference will be set. If None, reference will be generated based on the configuration.
        """
        if reference is None:
            if self.task_name == TaskName.REF_TRACKING:
                reference = np.random.uniform(
                    low=self.position_reference_low,
                    high=self.position_reference_high,
                )
            else:
                reference = np.random.uniform(
                    low=self.velocity_reference_low,
                    high=self.velocity_reference_high,
                )
        if len(reference) == 2:
            self.sim.velocity_reference = reference
        elif len(reference) == 3:
            self.sim.position_reference = reference
        else:
            raise ValueError("Unknown reference length.")

    def gym_masker(
        self, observability: Optional[List[str]] = None
    ):  # -> Callable[SegwaySimData, np.ndarray]:
        """Generates a function that will convert the observation to a np.array, possibly hiding some variables, as defined in the training config.

        Returns
        -------
        Callable[SegwaySimData, np.ndarray]
            The masking function.
        """
        if observability is None:
            observability = self.observability
        obs = SegwaySimData.empty()
        for field in observability:
            setattr(obs, field, 0)
        # TODO: this should preserve the order in task.observability
        mask = [f is not None for f in np.array(obs)]

        def to_gym(observation: SegwaySimData):
            return np.array(
                [v for v, observable in zip(np.array(observation), mask) if observable]
            )

        return to_gym

    def seed(self, seed):
        # for compatibility
        pass

    def close(self):
        """Close the render view window."""
        self.sim.close()

    def reset(
        self,
        seed: Optional[int] = None,
        state: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        reference: Optional[np.ndarray] = None,
        options: Dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Resets the environment (AI Gym interface method).

        Optionally, you can define (parts) of the new environment state.

        Parameters
        ----------
        state : Optional[np.ndarray], optional
        position : Optional[np.ndarray], optional
        reference : Optional[np.ndarray], optional

        Returns
        -------
        np.ndarray
            An observation after the reset.
        """
        super().reset(seed=seed)
        self.sim.reset(state, position)
        self.reward.reset()
        self.new_reference(reference)
        obs = self.sim.measure()
        info = {}
        if self.render_mode == "human":
            self.render()
        return obs.to_gym(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        """One step of the environment (AI Gym interface method).

        Parameters
        ----------
        action : np.ndarray
            Action on the wheels.

        Returns
        -------
        Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]
            Next observation, reward, whether the episode is finished, other info.
        """
        info = {}
        action = np.clip(action, -1, 1)
        measurement = self.sim.run(action, 1 / self.control_frequency)

        data = self.sim.data()
        reward = self.reward.compute(data, action)
        terminated = self.reward.terminated(data)
        truncated = self.reward.truncated(data)

        if self.render_mode == "human":
            self.render()
        return measurement.to_gym(), reward, terminated, truncated, info

    @property
    def data(self):
        return self.sim.measure()
