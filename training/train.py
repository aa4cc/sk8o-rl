import copy
import os
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np

# from stable_baselines3.common.logger import configure
import torch
import wandb
from callbacks import ActorToOnnx, RecordEpisode
from d2rl import D2RL_SACPolicy
from env.common import ActionFrameStack
from env.sk8o_full import SK8OFullEnv
from env.sk8o_segway import SegwayEnv
from gymnasium.wrappers.flatten_observation import FlattenObservation
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sbx import DroQ as DroQ_JAX
from stable_baselines3 import A2C, SAC, TD3, HerReplayBuffer
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback, EveryNTimesteps
from stable_baselines3.common.noise import NormalActionNoise
from tracking import LocalTracker, WandbTracker

SAC.policy_aliases["d2rl"] = D2RL_SACPolicy

def register_sk8o(cfg):
    # make the environments available to gym.make
    for env_name, entrypoint in (
        ("sk8o_segway", SegwayEnv),
        ("sk8o_full", SK8OFullEnv),
    ):
        for evaluation in (False, True):
            gym.register(
                f"{env_name}{'_eval' if evaluation else ''}",
                entrypoint,
                kwargs={
                    "env_cfg": cfg.env,
                    "task_cfg": cfg.task,
                    "evaluation": evaluation,
                },
            )


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def train(cfg: OmegaConf):
    replay_buffer = ReplayBuffer
    replay_buffer_kwargs = {}

    # initialize the environments
    register_sk8o(cfg)
    if cfg.env.name == "gym":
        env = lambda: gym.make(
            cfg.env.task, render_mode="human" if cfg.render_training else "rgb_array"
        )
        eval_env = lambda: gym.make(cfg.env.task, render_mode="rgb_array")
    else:
        env = lambda: gym.make(
            cfg.env.name, render_mode="human" if cfg.render_training else "rgb_array"
        )
        eval_env = lambda: gym.make(cfg.env.name + "_eval", render_mode="rgb_array")

    # wrap the envs in frame stack if necessary
    wrapper_classes = []
    wrapper_kwargses = []
    if cfg.frame_stacking.use:
        wrapper_classes.append(ActionFrameStack)
        wrapper_kwargses.append(
            {
                "num_stack": cfg.frame_stacking.length,
                "stack_action": cfg.frame_stacking.stack_action,
            }
        )
    if cfg.algorithm.name == "droq_jax":
        wrapper_classes.append(FlattenObservation)
        wrapper_kwargses.append({})
    # use vectorized environments to speed up training
    env = env()
    eval_env = eval_env()

    for wrapper, kwargs in zip(wrapper_classes, wrapper_kwargses):
        env = wrapper(env, **kwargs)
        eval_env = wrapper(eval_env, **kwargs)
    record_env = copy.deepcopy(eval_env)

    # algorithm setup
    if cfg.algorithm.name == "a2c":
        Trainer = A2C
    elif cfg.algorithm.name in ("sac", "d2rl"):
        Trainer = SAC
    elif cfg.algorithm.name in ("assisted_sac", "assisted_d2rl"):
        Trainer = AssistedSAC
    elif cfg.algorithm.name == "assisted_droq":
        Trainer = AssistedDroQ
    elif cfg.algorithm.name == "droq_jax":
        Trainer = DroQ_JAX
    elif cfg.algorithm.name == "td3":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )
        Trainer = lambda *args, **kwargs: TD3(
            *args, **kwargs, action_noise=action_noise
        )
    else:
        raise NotImplementedError("Unknown training algorithm.")

    if (
        cfg.pretrained_id is not None
        or cfg.algorithm.name == "assisted_sac"
        or cfg.expert_trajectories > 0
    ):
        # `learning_starts` works in a really stupid way in sb3, generating random actions instead of sampling from the agent, so setting a nonzero `learning_starts` causes detrimental action-observation pairs to be in the buffer, not those generated from the expert/pretrained policy -> make it zero
        cfg.learning_starts = 0
    if Trainer is not DroQ_JAX:
        model = Trainer(
            cfg.algorithm.policy.name,
            env,
            learning_rate=cfg.optimizer.lr,
            learning_starts=cfg.learning_starts,
            policy_kwargs={
                "net_arch": list(cfg.algorithm.policy.net_arch),
            },
            replay_buffer_class=replay_buffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            gradient_steps=cfg.algorithm.gradient_steps,
            verbose=1,
            target_entropy=cfg.algorithm.target_entropy,
            batch_size=cfg.algorithm.batch_size,
            tau=cfg.algorithm.tau,
            gamma=cfg.algorithm.gamma,
            device=cfg.device,
            seed=cfg.seed,
        )
    else:
        model = Trainer(
            cfg.algorithm.policy.name,
            env,
            policy_kwargs={
                "net_arch": list(cfg.algorithm.policy.net_arch),
            },
            dropout_rate=cfg.algorithm.dropout_rate,
            batch_size=cfg.algorithm.batch_size,
            tau=cfg.algorithm.tau,
            gamma=cfg.algorithm.gamma,
        )

    # potentially resume training
    if cfg.pretrained_id is not None:
        checkpoint, old_cfg = WandbTracker.best_checkpoint(cfg.pretrained_id)
        model.set_parameters(checkpoint, exact_match=True)
    if cfg.expert_trajectories > 0:
        model.replay_buffer = generate_trajectories(cfg, env(), model.replay_buffer)
    if cfg.algorithm.name in ("assisted_sac", "assisted_droq", "assisted_d2rl"):
        model.add_controller(eval_env.baseline_controller())
        model.action_overwrite_coeff(cfg.algorithm.aoc)

    # setup tracking
    output_dir = HydraConfig.get().runtime.output_dir
    if cfg.wandb:
        tracker = WandbTracker(cfg, output_dir)
    else:
        tracker = LocalTracker(cfg, output_dir)

    model.set_logger(tracker.sb3_logger())

    # define the callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(output_dir, "models"),
        callback_on_new_best=ActorToOnnx(env, tracker.output_dir, tracker=tracker),
        log_path=output_dir,
        eval_freq=cfg.evaluation.frequency,
        n_eval_episodes=cfg.evaluation.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks = [eval_callback]
    if cfg.video.use:
        record_callback = EveryNTimesteps(
            cfg.video.frequency,
            RecordEpisode(
                record_env,
                output_dir,
                cfg.video.fps,
                cfg.video.max_len,
                cfg.video.slow_motion_factor,
                cfg.task.control_frequency,
                cfg.video.folder,
                tracker,
            ),
        )
        callbacks.append(record_callback)

    # start training and save model at the end
    model.learn(total_timesteps=cfg.optimizer.total_timesteps, callback=callbacks)
    model.save_replay_buffer(Path(output_dir) / "final_buffer")
    tracker.finish()


if __name__ == "__main__":
    train()
