import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import imageio
import numpy as np
from env.sk8o_segway import SegwayEnv
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import BaseCallback
from tracking import Tracker
from utils import onnx_export


class ActorToOnnx(BaseCallback):
    def __init__(
        self,
        env: gym.Env,
        output_dir: str,
        filename: str = "sk8o_actor.onnx",
        tracker: Optional[Tracker] = None,
    ):
        self.env = env
        self.output_dir = Path(output_dir)
        self.filename = Path(output_dir) / filename
        self.n_calls = 0
        self.tracker = tracker

    def _on_step(self):
        model = self.parent.model
        model.save(self.output_dir / "best_checkpoint.zip")
        if self.tracker is not None:
            self.tracker.save_model(str(self.filename))
            self.tracker.save_model(str(self.output_dir / "best_checkpoint.zip"))
        onnx_export(model, self.env, self.filename)


class RecordEpisode(BaseCallback):
    """A callback that records a video of the environment. Called from SB3 training."""

    def __init__(
        self,
        env: SegwayEnv,
        output_dir: str,
        fps: int,
        max_len: float,
        slow_motion_factor: int,
        control_frequency: int,
        video_folder: str,
        tracker: Optional[Tracker] = None,
    ):
        """
        Parameters
        ----------
        env : SegwayEnv
            Environment to be recorded.
        output_dir : str
            Where to save the videos.
        cfg : OmegaConf
            The config defining other properties such as frequency, etc.
        """
        self.env = env
        self.output_dir = os.path.join(output_dir, video_folder)
        os.makedirs(self.output_dir)
        self.n_calls = 0
        self.tracker = tracker
        self.max_len = max_len
        self.fps = fps
        self.slow_motion_factor = slow_motion_factor
        self.control_frequency = control_frequency

    def _on_step(self):
        """Called by the EveryNTimesteps callback."""
        obs, info = self.env.reset()
        if self.env.render_mode != "rgb_array":
            try:
                self.env.render_mode = "rgb_array"
            except AttributeError:
                print("Could not log video.")
                return
        images = [self.env.render()]
        try:
            steps_per_frame = max(
                1, self.env.control_frequency // (self.fps * self.slow_motion_factor)
            )
        except AttributeError:
            # gym env
            steps_per_frame = 1
        for i in range(self.max_len * self.fps):
            for j in range(steps_per_frame):
                action, _ = self.model.predict(obs, deterministic=True)
                if action.shape[0] == 1 and len(action.shape) == 2:
                    action = action[0]
                obs, _, terminated, truncated, _ = self.env.step(action)
            im = self.env.render()
            images.append(im)
            if terminated or truncated:
                break
        video_path = f"{self.output_dir}/sample-{self.num_timesteps}steps.gif"
        imageio.mimwrite(video_path, images, duration=1000 / self.fps)
        if self.tracker is not None:
            self.tracker.save_video(video_path, self.num_timesteps)
        # if self.wandb:
        #     wandb.log({"eval_video": wandb.Video(video_path)}, )
        #     data = wandb.Table(
        #         data=np.concatenate(
        #             [
        #                 self.env.error_history(),
        #                 self.env.variable_history("phi")[:, [1]],
        #             ],
        #             axis=1,
        #         ),
        #         columns=["time", "error", "phi"],
        #     )
        #     wandb.log({"eval/data": data}, step=self.num_timesteps)
        self.env.reset()
