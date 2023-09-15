from dataclasses import asdict, astuple, dataclass, fields
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from gymnasium import spaces
from numpy.linalg import norm
from sk8o_sim import FullSimulationData, SK8OHipController, SK8OHipControllerCfg


class ObservationMode(Enum):
    """To avoid using strings when checking for mode, this enum is used."""

    SEGWAY = "segway"  # outputs (ẋ,φ̇,ψ̇,φ,ẋ_ref,ψ̇_ref)
    # the above plus hip motor positions, velocities and height reference
    SEGWAYPLUS = "segway+"
    HER = "her"
    SENSORS = "sensors"  # outputs the data from the sensors
    ALL = "all"  # outputs everything


class ActionMode(Enum):
    WHEELS = "wheels"
    BOTH = "both"  # hips and wheels
    AID = "aid"  # torque-controlled wheels and hips controlled by a PID


class SK8O_IO:
    """The class used as an interface between AI Gym and the simulation/reward generation, depending on the task setup."""

    def __init__(
        self,
        model,
        observation_mode: str,
        action_mode: str,
        measurement_noise_std: Optional[np.ndarray] = None,
    ):
        """

        Parameters
        ----------
        model : mujoco.MjModel
            The mujoco model
        observation_mode : str
            The observation mode, needs to define ObservationMode
        action_mode : str
            The action mode, needs to define ActionMode
        link_lengths : LinkLengths
            An object specifying the lenghts of all the links in SK8O.
        measurement_noise_std : np.ndarray
            A vector of measurement error standard deviations.
        """
        self.model = model
        self.observation_mode = ObservationMode(observation_mode)
        self.action_mode = ActionMode(action_mode)
        self.measurement_noise_std = measurement_noise_std

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

        if observation_mode is not None:
            if (
                self.observation_mode.value != observation_mode
                and self.measurement_noise_std is not None
            ):
                if (
                    self.observation_mode == ObservationMode.SEGWAYPLUS
                    and observation_mode == ObservationMode.SEGWAY.value
                ):
                    self.measurement_noise_std = np.concatenate(
                        [self.measurement_noise_std[:4], np.zeros(2)]
                    )
                else:
                    # no clear way to change the noise
                    print(
                        "Redefine noise after changing the observation mode: it will be set to zero now."
                    )
                    self.measurement_noise_std = None
            self.observation_mode = ObservationMode(observation_mode)
        if action_mode is not None:
            self.action_mode = ActionMode(action_mode)

    def change_control_frequency(self, control_frequency: float):
        ## set frame_skip according to the requested control frequency
        self.hip_controller = SK8OHipController(
            SK8OHipControllerCfg(control_frequency=control_frequency)
        )

    def gym_observation(
        self, data: FullSimulationData, use_noise: bool = True
    ) -> np.ndarray:
        """Returns and AI Gym observation as a numpy array according to the selecting observation mode.

        Returns
        -------
        np.ndarray
            The observation

        Raises
        ------
        NotImplementedError
            Unknown observation mode.
        """
        if self.observation_mode == ObservationMode.SEGWAY:
            gym_obs = data.segway_obs
        elif self.observation_mode == ObservationMode.SEGWAYPLUS:
            gym_obs = data.segwayplus_obs
        elif self.observation_mode == ObservationMode.HER:
            desired_goal = np.array(
                [data.dot_x_ref, data.dot_psi_ref, data.hip_angle_ref]
            )
            achieved_goal = np.array(
                [
                    data.dot_x,
                    data.dot_psi,
                    data.hip_angle_mean,
                ]
            )
            gym_obs = {
                "observation": data.her_obs,
                "achieved_goal": achieved_goal,
                "desired_goal": desired_goal,
            }
        elif self.observation_mode == ObservationMode.SENSORS:
            gym_obs = data.sensors_obs
        elif self.observation_mode == ObservationMode.ALL:
            gym_obs = data.full_obs
        else:
            raise NotImplementedError
        # this doesn't work in case of DictObs but HER is not documented anyways
        if (not use_noise) or self.measurement_noise_std is None:
            noise = 0 * gym_obs
        else:
            noise = self.measurement_noise_std * np.random.randn(*gym_obs.shape)
        if noise.shape != gym_obs.shape:
            raise ValueError(
                "Incompatible noise_std vector and obs_mode. Make sure they have the same length!"
            )
        return gym_obs + noise

    def observation_info(self) -> Tuple[List[str], Union[spaces.Box, spaces.Dict]]:
        """Returns a list of the names of the individual variables returned by gym observation, as well as the observation space.

        Returns
        -------
        Tuple[List[str],Union[spaces.Box,spaces.Dict]]
            Names and the space.

        Raises
        ------
        NotImplementedError
            On unknown observation mode.
        """
        # mock because the names are saved as a property and properties cannot be static in python
        mock_data = FullSimulationData(*([None] * 21))

        if self.observation_mode == ObservationMode.SEGWAY:
            names = mock_data.segway_names
        elif self.observation_mode == ObservationMode.SEGWAYPLUS:
            names = mock_data.segwayplus_names
        elif self.observation_mode == ObservationMode.HER:
            names = mock_data.her_names
        elif self.observation_mode == ObservationMode.SENSORS:
            names = mock_data.sensors_names
        elif self.observation_mode == ObservationMode.ALL:
            names = mock_data.all_names
        else:
            raise NotImplementedError
        if self.observation_mode != ObservationMode.HER:
            observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(names),),
                dtype=np.float64,
            )
            return names, observation_space
        else:
            observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(names),),
                dtype=np.float64,
            )
            goal_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),
                dtype=np.float64,
            )
            return names, spaces.Dict(
                {
                    "observation": observation_space,
                    "achieved_goal": goal_space,
                    "desired_goal": goal_space,
                }
            )

    def reset(self):
        self.hip_controller.reset()

    def action(self, action: np.ndarray, data: FullSimulationData) -> np.ndarray:
        """Parses the action generated by the agent based on the action mode.

        Parameters
        ----------
        action : np.ndarray
            The action generated by the agent.

        Returns
        -------
        np.ndarray
            A 4-vector of action on each of the four motors.
        """
        if self.action_mode == ActionMode.WHEELS:
            action = np.concatenate([[0, 0], action])
        elif self.action_mode == ActionMode.AID:
            # overwrite the position reference by toqrue according to the PID
            hip_action = self.hip_controller.action(data)
            action = np.concatenate([hip_action, action])
        return action

    def action_info(self) -> Tuple[List[str], Union[spaces.Box, spaces.Dict]]:
        """Returns a list of the names of the individual variables that are expected from the agent, as well as the action space.

        Returns
        -------
        Tuple[List[str],Union[spaces.Box,spaces.Dict]]
            Names and the space.

        Raises
        ------
        NotImplementedError
            On unknown action mode.
        """
        if self.action_mode == ActionMode.WHEELS:
            action_space = spaces.Box(low=-1, high=1, shape=(2,))
            action_names = [
                self.model.actuator(i).name for i in range(2, self.model.nu)
            ]
        elif self.action_mode == ActionMode.AID:
            from .sk8o_full_env import SK8OFullEnv

            hip_min = SK8OFullEnv.hip_angle_range[0]
            hip_max = SK8OFullEnv.hip_angle_range[1]
            action_space = spaces.Box(
                low=np.array([hip_min, hip_min, -1, -1]),
                high=np.array([hip_max, hip_max, 1, 1]),
                shape=(4,),
            )
            # information about observations and action spaces
            action_names = [
                self.model.actuator(i).name for i in range(2, self.model.nu)
            ]
        elif self.action_mode == ActionMode.BOTH:
            action_space = spaces.Box(low=-1, high=1, shape=(4,))
            # information about observations and action spaces
            action_names = [self.model.actuator(i).name for i in range(self.model.nu)]
        return action_names, action_space

    def info(self, data: FullSimulationData) -> Dict[str, Dict[str, float]]:
        """Given SB3's implementation of HER, it is necessary to include the observation in info to be able to recompute the reward with a new goal.

        Returns
        -------
        Dict[str,Dict[str,float]]
            The FullSimulationData dataclass saved as a dictionary.
        """
        if self.observation_mode == ObservationMode.HER:
            return {"full_obs": asdict(data)}
        else:
            return {}

    def info2obs(
        self, info: Dict[str, Dict[str, float]], desired_goal
    ) -> FullSimulationData:
        """Recreate the observation from the dict generated by `info` above with the reference changed.

        Parameters
        ----------
        info : Dict[str, Dict[str, float]]
            The info dict generated by the function above.
        desired_goal : np.ndarray
            The changed reference.

        Returns
        -------
        FullSimulationData
            The observation extracted from the dict.
        """
        obs = FullSimulationData.from_dict(info["full_obs"])
        obs.dot_x_ref, obs.dot_psi_ref, obs.hip_angle_ref = desired_goal
        return obs
