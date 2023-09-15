from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from omegaconf import OmegaConf
from sk8o_sim import SegwaySimData

from ..common import TaskName


class Reward:
    """This class keeps track of the task settings.

    This includes generating the configured initial conditions and generating step cost for training.
    """

    def __init__(self, task_cfg: OmegaConf, evaluation: bool):
        """
        Parameters
        ----------
        task_cfg : OmegaConf
        evaluation : bool
            If true, the evaluation cost in task_cfg will be used.
        """
        self.name = TaskName(task_cfg.name)
        self.observability = task_cfg.observability
        self._warned_about_distance = False
        self.control_frequency = task_cfg.control_frequency

        if evaluation:
            task_cfg = task_cfg.eval
        else:
            task_cfg = task_cfg.train
        self.cost_cfg = task_cfg.cost
        if evaluation:
            # make evaluation scores per episode independent of control frequency
            for k, _ in self.cost_cfg.items():
                if k not in ["fall", "goal_reached"]:
                    self.cost_cfg[k] *= 50 / self.control_frequency
        self.end_conditions = task_cfg.end_conditions
        self.reference_cfg = task_cfg.reference
        self.reset()

    def reference_error(self, obs: SegwaySimData) -> float:
        """Returns the Euclidian norm of the reference error, based on the task.

        Parameters
        ----------
        obs : SegwaySimData
            Current environment observation.

        Returns
        -------
        float
            Reference error norm.

        Raises
        ------
        ValueError
            When no reference error is set.
        """
        if self.name == TaskName.BALANCE:
            return np.linalg.norm(obs.velocity)
        elif obs.position_reference is None and obs.velocity_reference is None:
            raise ValueError("You need to set a reference first!")
        if self.name == TaskName.REF_TRACKING:
            diff = np.array(obs.position) - np.array(obs.position_reference)
            return np.linalg.norm(diff[:2])  # ignore rotation (for now)
        elif self.name == TaskName.MOTION_CONTROL or self.name == TaskName.GO_FORWARD:
            diff = [obs.dot_x - obs.dot_x_ref, obs.dot_psi - obs.dot_psi_ref]
            return np.linalg.norm(diff)
        else:
            if not self._warned_about_distance:
                print(
                    "Calculating target distance but Task is not 'reference tracking' nor 'motion control': returning 0. This warning will only appear once."
                )
                self._warned_about_distance = True
            return 0

    def _not_moving(self, obs: SegwaySimData) -> bool:
        return np.linalg.norm(obs.velocity) < self.end_conditions.reference_derivative

    def _not_accelerating(self, obs: SegwaySimData) -> bool:
        return (
            np.linalg.norm(self.acceleration_running_avg)
            < self.end_conditions.reference_derivative
        )

    def _goal_reached(self, obs: SegwaySimData) -> bool:
        """Decide whether the observation fulfills the goal defined in the config.

        Parameters
        ----------
        obs : SegwaySimData
            Current environment observation.

        Returns
        -------
        bool
            Whether the goal is reached.
        """
        if self.name == TaskName.MOTION_CONTROL or self.name == TaskName.GO_FORWARD:
            close_enough = (
                self.reference_error(obs) < self.end_conditions.reference_error
            )
            historically_close_enough = (
                self.reference_error_running_avg < self.end_conditions.reference_error
            )
            # print(
            #     self.reference_error(obs),
            #     self.reference_error_running_avg,
            #     self._not_accelerating(obs),
            #     self.end_conditions.reference_error,
            # )
            return historically_close_enough and self._not_accelerating(obs)
        elif self.name == TaskName.REF_TRACKING:
            close_enough = (
                self.reference_error(obs) < self.end_conditions.reference_error
            )
            return close_enough and self._not_moving(obs)
        else:
            return False

    def _tipped_over(self, obs: SegwaySimData) -> bool:
        # returns true if the inclination angle is out of the allowed range
        return abs(obs.phi) > self.end_conditions.phi

    def truncated(self, obs: SegwaySimData) -> bool:
        """Decide, whether the training epoch should end due to time limit.

        Parameters
        ----------
        obs : SegwaySimData
            Current environment observation.

        Returns
        -------
        bool
            True if time limit reached.
        """
        too_long = obs.time > self.end_conditions.time
        return too_long

    def terminated(self, obs: SegwaySimData) -> bool:
        """Decide, whether the training epoch should end due to final state being reached.

        Parameters
        ----------
        obs : SegwaySimData
            Current environment observation.

        Returns
        -------
        bool
            True if final state reached.
        """
        return self._tipped_over(obs) or self._goal_reached(obs)

    def reset(self):
        """To be called on env reset."""
        self.last_t = 0
        # make it so it's impossible to finish within the first 0.2s (this is an arbitrary number)
        self.acceleration_running_avg = np.power(
            self.end_conditions.reference_derivative, 1 / (self.control_frequency * 0.2)
        )
        self.reference_error_running_avg = np.power(
            self.end_conditions.reference_error, 1 / (self.control_frequency * 0.2)
        )

    def compute(self, obs: SegwaySimData, action: np.ndarray) -> float:
        """Calculates the cost of the obervation and action based on the config.

        Parameters
        ----------
        obs : SegwaySimData
            Current environment observation.
        action : np.ndarray
            Action in this step.

        Returns
        -------
        float
            The cost.
        """

        def diagonal_quadratic(A, v):
            return np.transpose(v) @ np.diag(A) @ v

        max_phi_cost = 1
        max_input_cost = 2
        max_error_cost = np.sqrt(8)  # not true for ref tracking, there it's infinite...

        cost = (
            self.cost_cfg.phi * obs.phi**2 / max_phi_cost
            + np.dot(action, action) * self.cost_cfg.wheel_input / max_input_cost
        )

        cost += self.cost_cfg.step
        if obs.time < self.last_t:
            self.reset()
        elif obs.time > self.last_t:
            if obs.acceleration is not None:
                self.acceleration_running_avg = (
                    0.95 * self.acceleration_running_avg + 0.05 * obs.acceleration
                )
            if self.reference_error_running_avg is not None:
                self.reference_error_running_avg = (
                    0.95 * self.reference_error_running_avg
                    + 0.05 * self.reference_error(obs)
                )
            # print(
            #     self.acceleration_running_avg,
            #     np.linalg.norm(self.acceleration_running_avg),
            # )
            self.last_t = obs.time

        # task specific costs
        if self.cost_cfg.quadratic_errors:
            cost += (
                self.cost_cfg.error
                * self.reference_error(obs) ** 2
                / max_error_cost**2
            )
        else:
            cost += self.cost_cfg.error * self.reference_error(obs) / max_error_cost

        # cost for ending episode
        if self._goal_reached(obs):
            cost += self.cost_cfg.goal_reached  # negative value
        if self._tipped_over(obs):
            cost += self.cost_cfg.fall
        return -cost
