from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf
from sk8o_sim import FullSimulation, FullSimulationCfg, FullSimulationData

from .io import SK8O_IO
from .reward import Reward


class SK8OFullEnv(gym.Env):
    """This is the main class to be used to train RL agents on SK8O's MuJoCo simulation."""

    def reset_model(self):
        # defined abstract TODO:
        pass

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
        "observation_modes": [
            "segway",  # outputs (ẋ,φ̇,ψ̇,φ,ẋ_ref,ψ̇_ref)
            "segway+",  # the above plus motor positions, velocities and height reference
            "sensors",  # outputs the data from the sensors
            "all",  # outputs everything
        ],
    }

    leg_length_range = (120e-3, 330e-3)
    hip_angle_range = (0.31, 1.07)  # slightly larger to allow for soft limits
    alpha0 = np.pi / 4

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

        task_cfg : OmegaConf

        evaluation : bool, optional
            True if this is to be an evaluation environment, by default False
        render_mode : Optional[str], optional
            Render mode as defined by the parent class MujocoEnv, by default None
        """

        measurement_noise_std = np.array(env_cfg.simulation.measurement_noise_std)
        env_cfg.simulation.__delattr__("measurement_noise_std")
        self.sim = FullSimulation(FullSimulationCfg(**env_cfg.simulation))
        self.io = SK8O_IO(
            self.sim.robot.model,
            env_cfg.io.obs_mode,
            env_cfg.io.action_mode,
            measurement_noise_std if env_cfg.simulation.use_noise else None,
        )

        self.change_control_frequency(task_cfg.control_frequency)
        self.change_mode(env_cfg.io.obs_mode, env_cfg.io.action_mode)

        task_name = task_cfg.name
        if evaluation:
            task_cfg = task_cfg.eval
        else:
            task_cfg = task_cfg.train
        self.reference_cfg = task_cfg.reference
        self.reward = Reward(task_cfg, task_name, self.control_frequency, evaluation)

    def change_control_frequency(self, control_frequency: float):
        self.control_frequency = control_frequency
        self.io.change_control_frequency(control_frequency)

    @property
    def render_mode(self) -> str:
        self.sim.view.mode

    @render_mode.setter
    def render_mode(self, mode: str):
        self.sim.view.mode = mode

    def new_reference(
        self,
        velocity_reference: Tuple[float, float] | None = None,
        hip_angle_reference: float | None = None,
    ):
        """Sets the reference that will be included in the data from now on.

        Parameters
        ----------
        velocity_reference : Tuple[float, float] | np.ndarray | None, default None.
            The velocity_reference (dot_x_ref, dot_psi_ref). If None, reference will be generated based on the configuration.
        hip_angle_reference : float, optional
            Distance from the ground (assumes upright orientation). If None, reference will be generated based on the configuration. By default None
        """
        # reference = self.task.new_reference(np.array(reference))

        if velocity_reference is None:
            velocity_reference = np.random.uniform(
                low=self.reference_cfg.velocity.low,
                high=self.reference_cfg.velocity.high,
            )
        if hip_angle_reference is None:
            hip_angle_reference = np.random.uniform(
                low=self.sim.robot.height2hip_angle(self.reference_cfg.height.low),
                high=self.sim.robot.height2hip_angle(self.reference_cfg.height.high),
            )
        self.sim.new_reference(velocity_reference, hip_angle_reference)

    def compute_reward(
        self,
        achieved_goal: List[np.ndarray],
        desired_goal: List[np.ndarray],
        info: List[Dict[str, Dict[str, float]]],
    ) -> List[float]:
        """Recomputes the reward for a new goal. This method is used by HER. The function assumes vector input.

        Parameters
        ----------
        achieved_goal : List[np.ndarray]
            The goal that was achieved.
        desired_goal : List[np.ndarray]
            The goal that is the reference goal.
        info : List[Dict[str, Dict[str, float]]]
            The info that contains the action and the FullSimulationData in the form of a dict, generated when using HER obs_mode.

        Returns
        -------
        List[float]
            The rewards.
        """
        rewards = []
        for a, d, i in zip(achieved_goal, desired_goal, info):
            self.her_task.reset()
            self.her_task.update_obs(self.io.info2obs(i, d))
            rewards.append(self.her_task.reward(i["action"]))
        return np.array(rewards)

    def reset(
        self,
        seed: Optional[int] = None,
        leg_lengths: Optional[Union[float, Tuple[float, float]]] = None,
        hip_angles: Optional[Union[float, Tuple[float, float]]] = None,
        velocity_reference: Tuple[float, float] | None = None,
        hip_angle_reference: float | None = None,
    ) -> np.ndarray:
        """A standard AI Gym API reset function with an additional option to set the leg lengths/hip angles and reference.

        Parameters
        ----------
        seed : Optional[int], optional
            The seed for random number generator, by default None
        leg_lengths : Optional[Union[float, Tuple[float, float]]], optional
            Lengths of legs, ignored if `hip_angles` not None, by default None
        hip_angles : Optional[Union[float, Tuple[float, float]]], optional
            The angles of the hips (alpha_1), by default None
        reference : Optional[np.ndarray], optional
            The reference, by default None

        Returns
        -------
        np.ndarray
            An observation.
        """
        super().reset(seed=seed)
        self.sim.reset(
            hip_angles, leg_lengths, 0, velocity_reference, hip_angle_reference
        )
        self.new_reference(velocity_reference, hip_angle_reference)
        data = self.sim.get_data()
        info = self.io.info(data) | {"action": np.array([0, 0, 0, 0])}

        if self.render_mode == "human":
            self.render()
        return self.io.gym_observation(data), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """The standard AI Gym API step function.

        Parameters
        ----------
        action : np.ndarray
            The action to be performed.

        Returns
        -------
        Tuple[np.ndarray, float, bool, bool, dict]
            Observation, rewrad, terminated, truncated, info
        """
        action = self.io.action(action, self.sim.get_data())
        data = self.sim.run(action, 1 / self.control_frequency)

        if self.render_mode == "human":
            self.render()
        return (
            self.obs(),
            *self.reward.compute(action, self.sim, data),
            self.io.info(data) | {"action": action},
        )

    def full_obs(self) -> FullSimulationData:
        # returns the FullSimulationData dataclass
        return self.sim.get_data().full_obs

    def segway_obs(self) -> np.ndarray:
        """Returns a segway obs (can be used with the LQR controller)

        Returns
        -------
        np.ndarray
            (dot_x,dot_phi, dot_psi, phi, dot_x_ref, dot_psi_ref)
        """
        return self.sim.get_data().segway_obs

    def obs(self) -> np.ndarray:
        # returns the gym obs based on the obs_mode selected
        return self.io.gym_observation(self.sim.get_data())

    def segway_mode(self, flag: bool = True, observability: Optional[OmegaConf] = None):
        """Changes the action and observation mode to be compatible with segway controllers.

        Parameters
        ----------
        flag : bool, optional
            If false, the actions and observations will be raw (distables segway mode), by default True
        """
        obs_mode = "segway" if flag else "sensors"
        action_mode = "wheels" if flag else "both"
        self.change_mode(obs_mode, action_mode)

    def change_mode(
        self,
        observation_mode: Optional[str] = None,
        action_mode: Optional[str] = None,
    ):
        """Changes the observations generated and/or the actions accepted by the system.

        Parameters
        ----------
        observation_mode : Optional[str], optional
            If not None, changes the observation mode, by default None
        action_mode : Optional[str], optional
            If not None, changes the action mode, by default None
        """
        self.io.change_mode(observation_mode, action_mode)
        self.action_names, self.action_space = self.io.action_info()
        self.observation_names, self.observation_space = self.io.observation_info()

    @property
    def data(self) -> FullSimulationData:
        return self.sim.get_data()

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

    @property
    def render_mode(self):
        return self.sim.view.mode

    @render_mode.setter
    def render_mode(self, value: str):
        self.sim.view.mode = value
