import copy
from copy import deepcopy
from dataclasses import asdict
from typing import List, Optional, Tuple

import gymnasium as gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
from env.common import ActionFrameStack
from env.sk8o_full import SK8OFullEnv
from env.sk8o_segway import SegwayEnv
from omegaconf import OmegaConf
from sk8o_sim.controllers import (
    SegwayLQRController,
    SegwayLQRControllerCfg,
    SK8OController,
    SK8OFullController,
    SK8OHipController,
    SK8OHipControllerCfg,
)
from tracking import WandbTracker


class OnnxController:
    def __init__(self, cfg: OmegaConf, model_path: str):
        self.cfg = cfg
        self.env_name = cfg.env.name
        self.sess = ort.InferenceSession(model_path)

    def predict(self, env, obs):
        obs = np.array(obs).astype(np.float32)
        return self.sess.run(None, {self.sess.get_inputs()[0].name: obs})[0]


class LQRController:
    def __init__(self, env_name: str, controller: SK8OController):
        self.env_name = env_name
        self.cfg = OmegaConf.create(
            {
                "frame_stacking": {"use": False},
                "env": {"obs_mode": "segway", "action_mode": "both"},
                "task": {
                    "observability": [
                        "dot_x",
                        "dot_phi",
                        "dot_psi",
                        "x",
                        "phi",
                        "psi",
                        "py",
                        "px",
                        "x_ref",
                        "y_ref",
                        "psi_ref",
                        "dot_x_ref",
                        "dot_psi_ref",
                    ]
                },
            }
        )
        self.controller = controller

    def predict(self, env, obs):
        return self.controller.action(env.data)

    def reset(self):
        self.controller.reset()


class SK8O_Eval:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self._env_setup(cfg)
        self.baseline_controller = self.get_baseline_controller(self.env)
        self.initial_conditions(None, None, None, None)

    def _env_setup(self, cfg: OmegaConf):
        if cfg.env.name == "sk8o_segway":
            env = SegwayEnv(cfg.env, cfg.task, evaluation=False)
        elif cfg.env.name == "sk8o_full":
            env = SK8OFullEnv(cfg.env, cfg.task, evaluation=False)
        else:
            raise ValueError("Unknown environment")
        self.env = env
        self.env_name = cfg.env.name

    def get_baseline_controller(
        self, env: SegwayEnv | SK8OFullEnv, control_frequency: float = 50
    ) -> LQRController:
        wheel_cfg = SegwayLQRControllerCfg(control_frequency=control_frequency)
        wheel_controller = SegwayLQRController(wheel_cfg)
        if isinstance(env, SegwayEnv):
            controller = wheel_controller
            env_name = "sk8o_segway"
        elif isinstance(env, SK8OFullEnv):
            hip_cfg = SK8OHipControllerCfg(control_frequency=control_frequency)
            controller = SK8OFullController(
                SK8OHipController(hip_cfg), wheel_controller
            )
            env_name = "sk8o_full"
        return LQRController(env_name, controller)

    def change_env(self, env_name: str, *overrides: Tuple[str]):
        # old_task_setup = self.cfg.task
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base="1.2", config_path="config")
        self.cfg = hydra.compose(
            config_name="config.yaml",
            overrides=[f"env={env_name}", f"task={self.cfg.task.name}", *overrides],
        )
        # TODO: merge default task cfg with the current one to keep rewards the same
        self._env_setup(self.cfg)
        # OmegaConf.update(self.cfg, "task", old_task_setup, force_add=True)
        self.baseline_controller = self.get_baseline_controller(self.env)

    def load_onnx(self, model_path: str):
        self.onnx_controller = OnnxController(self.cfg, model_path)

    @classmethod
    def from_wandb(cls, experiment_url: str):
        onnx_model, cfg = WandbTracker.experiment_result(experiment_url)
        eval = cls(cfg)
        eval.load_onnx(onnx_model)
        return eval

    def initial_conditions(
        self,
        state: np.ndarray | None = None,
        reference: np.ndarray | None = None,
        velocity_reference: Tuple[float, float] | None = None,
        height_reference: float | None = None,
    ):
        self.state0 = state
        self.reference0 = reference
        self.velocity_reference0 = velocity_reference
        self.height_reference0 = height_reference

    def _reset_env(self, env: Optional[gym.Env] = None):
        if env is None:
            env = self.env
        if isinstance(env, ActionFrameStack):
            test_env = env.env
        else:
            test_env = env
        if isinstance(test_env, SegwayEnv):
            obs, _ = env.reset(state=self.state0, reference=self.reference0)
        elif isinstance(test_env, SK8OFullEnv) or True:
            obs, _ = env.reset(
                velocity_reference=self.velocity_reference0,
                hip_angle_reference=None
                if self.height_reference0 is None
                else test_env.sim.robot.height2hip_angle(self.height_reference0),
            )
        else:
            raise ValueError("Unknown environment")
        return obs

    def _run(
        self,
        controller,
        max_time: float,
        stop_on_termination: bool,
        control_frequency: float = 1000,
        variable_reference: Optional[List[Tuple[float, List[Tuple]]]] = None,
        seed: Optional[int] = 42,
    ) -> Tuple[pd.DataFrame, List[float]]:
        # setup a few bools for later branching (yes, it's ugly and complicated but there's too many possible combinations)
        if seed is not None:
            np.random.seed(seed)
        segway_ctrl_full_env = (
            controller.env_name == "sk8o_segway" and self.env_name == "sk8o_full"
        )
        if (
            self.env_name == "sk8o_segway"
            and self.onnx_controller.env_name == "sk8o_full"
        ):
            raise ValueError(
                "Incompatible combination: Full controller on segway simulation!"
            )
        add_frame_stacking = controller.cfg.frame_stacking.use
        # possibly modify the environment to make it compatible with the controller
        env = copy.deepcopy(self.env)
        prev_cf = self.env.control_frequency
        env.change_control_frequency(control_frequency)
        # env = self.env
        if add_frame_stacking:
            env = ActionFrameStack(
                env,
                controller.cfg.frame_stacking.length,
                controller.cfg.frame_stacking.stack_action,
            )
        # in case the controller needs to access the variables directly TODO: should not happen
        controller.env = env

        if self.env_name == "sk8o_segway":
            env.change_observability(controller.cfg.task.observability)
        elif self.env_name == "sk8o_full":  # and not segway_ctrl_full_env:
            env.change_mode(
                observation_mode=controller.cfg.env.get("obs_mode", "segway"),
                action_mode=controller.cfg.env.get("action_mode", "aid"),
            )
        if variable_reference is not None:
            self.reference0 = variable_reference[0][1]
        obs = self._reset_env(env)

        # now finally get to the test :-)
        history = []
        rewards = []
        terminated = False
        for t in np.arange(0, max_time, 1 / env.control_frequency):
            # potentially change reference
            self._vary_reference(env, variable_reference, t)
            action = controller.predict(env, obs)
            if add_frame_stacking:
                # only put the last observation in the history
                obs = env.last_observation()
            if self.env_name == "sk8o_segway":
                obs = np.concatenate([[env.data.time], obs])
            elif self.env_name == "sk8o_full":
                obs = np.concatenate([[t], obs])
            history.append([*obs, *action])
            if terminated and stop_on_termination:
                break
            obs, reward, terminated, truncated, info = env.step(action)
            # if terminated:
            #     print("fallen")
            # elif truncated:
            #     print("survived")
            #     break
            rewards.append(reward)
        env.change_control_frequency(prev_cf)
        columns = [
            "time",
            *env.observation_names,
            *env.action_names,
        ]
        print(columns)
        # if self.env_name == "sk8o_segway":
        # elif self.env_name == "sk8o_full":
        #     columns = [*env.action_names, env.observation_names)]
        # else:
        #     raise NotImplementedError("Unknown env!")
        return pd.DataFrame(history, columns=columns), rewards

    def _vary_reference(self, env, variable_reference, curr_time):
        if variable_reference is not None:
            reference = next(
                (
                    reference
                    for (time, reference) in variable_reference[::-1]
                    if curr_time >= time
                ),
                variable_reference[0][1],
            )
            # print(reference)
            if self.env_name in ("sk8o_segway",):
                env.new_reference(reference)
            elif self.env_name == "sk8o_full":
                raise NotImplementedError(
                    "Variable reference currently only available for segway"
                )
            else:
                raise NotImplementedError(
                    "Variable reference not supported for this environment."
                )

    def baseline(
        self,
        max_time: float = 10,  # [s]
        stop_on_termination: bool = True,
        control_frequency: float = 1000,
        variable_reference: Optional[List[Tuple[float, List[Tuple]]]] = None,
    ):
        # variable reference: a sorted list of times and references
        self.baseline_controller = self.get_baseline_controller(
            self.env, control_frequency=control_frequency
        )
        self.baseline_controller.reset()
        ret = self._run(
            self.baseline_controller,
            max_time,
            stop_on_termination,
            control_frequency,
            variable_reference,
        )

        return ret

    def onnx(
        self,
        max_time: float = 10,  # [s]
        stop_on_termination: bool = True,
        onnx_controller: Optional[OnnxController] = None,
        control_frequency: float = 50,
        variable_reference: Optional[List[Tuple[float, List[Tuple]]]] = None,
    ):
        # variable reference: a sorted list of times and references
        if onnx_controller is None:
            onnx_controller = self.onnx_controller
        ret = self._run(
            onnx_controller,
            max_time,
            stop_on_termination,
            control_frequency,
            variable_reference,
        )
        return ret

    def render(self, render: bool = True):
        if render:
            print(self.env, self.env.render_mode)
            self.env.render_mode = "human"
        else:
            self.env.render_mode = "rgb_array"

    def plot(self, history: pd.DataFrame, *columns):
        plt.figure()
        for c in columns:
            try:
                # print(history["time"].shape, history.loc[:, c].shape)
                plt.plot(history["time"], history.loc[:, c])
                # plt.plot(history.loc[:, c])
            except KeyError:
                print(f"Skipping column {c}, available are {history.columns.values}")
        plt.grid()
        plt.legend(columns)
        plt.xlabel("time [s]")
