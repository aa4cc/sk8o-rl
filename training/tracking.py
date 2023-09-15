from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common import logger as sb3logger
from stable_baselines3.common.logger import configure

# from ..utils import ActionFrameStack


class Tracker:
    @classmethod
    def _cfg_merge(cls, experiment_cfg: DictConfig) -> DictConfig:
        # load a default cfg for this task-env combo, in case the config has changed since the experiment was peformed
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base="1.2", config_path="config")
        task_name = (
            experiment_cfg.task.name
            if (experiment_cfg.env.name != "gym")
            else experiment_cfg.env.task
        )
        cfg = hydra.compose(
            config_name="config.yaml",
            overrides=[
                f"env={experiment_cfg.env.name}",
                f"task={task_name}",
            ],
        )
        OmegaConf.update(
            cfg, "frame_stacking", experiment_cfg.frame_stacking, force_add=True
        )
        OmegaConf.update(cfg, "env", experiment_cfg.env, force_add=True)
        OmegaConf.update(cfg, "task", experiment_cfg.task, force_add=True)
        return cfg

    def experiment_result(self, run_path: str) -> Tuple[gym.Env, str]:
        # load experiment using the API, returns a generated env and path to .onnx model
        raise NotImplementedError

    def save_model(self, model_path: str):
        raise NotImplementedError

    def save_video(self, video_path: str, num_timesteps: int):
        raise NotImplementedError

    def sb3_logger(self) -> sb3logger.Logger:
        raise NotImplementedError

    def finish(self):
        raise NotImplementedError


class LocalTracker(Tracker):
    def __init__(self, cfg: DictConfig, output_dir: str):
        self.output_dir = output_dir
        self.cfg = cfg
        # yaml.dump(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True), self.output_dir)

    def sb3_logger(self) -> sb3logger.Logger:
        return configure(self.output_dir, ["stdout", "csv", "tensorboard"])

    def save_model(self, model_path: str):
        pass

    def save_video(self, video_path: str, num_timesteps: int):
        pass

    def finish(self):
        pass


class WandbWriter(sb3logger.KVWriter, sb3logger.SeqWriter):
    def write_sequence(self, sequence: List) -> None:
        """
        write_sequence an array to file

        :param sequence:
        """
        print(sequence)

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:
        """
        Write a dictionary to file

        :param key_values:
        :param key_excluded:
        :param step:
        """
        # print(key_excluded)
        wandb.log(key_values, step=step)

    def close(self) -> None:
        """
        Close owned resources
        """
        # raise NotImplementedError
        pass


class WandbTracker(Tracker):
    def __init__(self, cfg: DictConfig, output_dir: str) -> None:
        # init the tracking connection
        self.cfg = cfg
        task_name = cfg.task.name if (cfg.env.name != "gym") else cfg.env.task
        self.run = wandb.init(
            project=f"{cfg.env.name}-{task_name}".replace(" ", "_"),
            name=cfg.name,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            # sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            # monitor_gym=True,  # auto-upload the videos of agents playing the game
            group=cfg.group,
            save_code=True,  # optional
            dir=output_dir,
        )
        self.output_dir = wandb.run.dir

    @classmethod
    def experiment_result(cls, run_path: str) -> Tuple[str, DictConfig]:
        # load experiment using the API, returns a path to .onnx model and the original config
        api = wandb.Api()
        run = api.run(run_path)
        experiment_cfg = OmegaConf.create(run.config)
        cfg = cls._cfg_merge(experiment_cfg)
        model_path = f"onnx_models/{cfg.env.name}/{cfg.task.name}/{run._attrs['name']}"
        model_name = "sk8o_actor.onnx"
        run.file(model_name).download(root=model_path, exist_ok=True)
        return f"{model_path}/{model_name}", cfg

    @classmethod
    def best_checkpoint(cls, run_path: str) -> str:
        api = wandb.Api()
        run = api.run(run_path)
        experiment_cfg = OmegaConf.create(run.config)
        cfg = cls._cfg_merge(experiment_cfg)
        model_path = (
            f"previous_runs/{cfg.env.name}/{cfg.task.name}/{run._attrs['name']}"
        )
        model_name = "best_checkpoint.zip"
        run.file(model_name).download(root=model_path, exist_ok=True)
        return f"{model_path}/{model_name}", cfg

    def save_model(self, model_path: str):
        wandb.save(model_path)

    def save_video(self, video_path: str, num_timesteps: int):
        wandb.log({"eval_video": wandb.Video(video_path)}, step=num_timesteps)

    def sb3_logger(self) -> sb3logger.Logger:
        writer = WandbWriter()
        return sb3logger.Logger(self.output_dir, output_formats=[writer])

    def finish(self):
        return self.run.finish()
