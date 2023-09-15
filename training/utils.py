import os
import sys
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import onnxruntime as ort
import torch
from env.sk8o_segway import SegwayEnv
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStack
from omegaconf import OmegaConf
from stable_baselines3 import A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from sbx import DroQ
from torch import nn
from typing import Iterable


def get_model_file(experiment_folder: Path, filename="best_model.zip"):
    files = [f for f in (experiment_folder / "models").iterdir()]
    candidates = [f for f in files if filename in str(f)]
    if len(candidates) == 0:
        raise FileNotFoundError(f"Model not found in {experiment_folder}")
    elif len(candidates) > 1:
        raise ValueError(f"Multiple models found, IDK which one to choose. :-(")
    else:
        return candidates[0]


def reload_from_history(experiment_folder):
    cfg = OmegaConf.load(experiment_folder / ".hydra" / "config.yaml")
    env = SegwayEnv(cfg.env, cfg.task)

    model_file = get_model_file(experiment_folder)
    if cfg.algorithm.name == "a2c":
        model = A2C.load(model_file, env, device="cpu")
    elif cfg.algorithm.name == "sac":
        model = SAC.load(model_file, env, device="cpu")
    else:
        raise NotImplementedError("Unknown training algorithm.")
    return env, model


class OnnxablePolicy(torch.nn.Module):
    def __init__(self, actor: torch.nn.Module):
        super().__init__()
        # Removing the flatten layer because it can't be onnxed
        self.actor = torch.nn.Sequential(
            actor.latent_pi,
            actor.mu,
            # For gSDE
            # torch.nn.Hardtanh(min_val=-actor.clip_mean, max_val=actor.clip_mean),
            # Squash the output
            torch.nn.Tanh(),
            # TODO: multiply by range? this only works because env has clip +-1
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # NOTE: You may have to process (normalize) observation in the correct
        #       way before using this. See `common.preprocessing.preprocess_obs`
        observation = torch.flatten(observation)
        return torch.squeeze(self.actor(observation))

class OnnxableDroQJAX(nn.Module):
    def __init__(self, observation_dim:int, action_dim:int, net_arch:Iterable[int]=[128,128]):
        super().__init__()
        last_dim=observation_dim
        layers=[]
        for d in net_arch:
            layers.append(nn.Linear(last_dim,d))
            layers.append(nn.ReLU())
            last_dim=d
        self.common=nn.Sequential(*layers)
        # don't care about the std - this will be used for inference
        self.mu=nn.Linear(last_dim,action_dim)
        self.action=nn.Tanh()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        observation = torch.flatten(observation)
        x=self.common(observation)
        mu=self.mu(x)
        action=self.action(mu)
        return torch.squeeze(action)

    @classmethod
    def from_sbx(cls, model:DroQ):
        n_layers=len(model.policy.actor_state.params["params"])
        params=[]
        for i in range(n_layers-1):# last layer is std
            layer=model.policy.actor_state.params["params"][f"Dense_{i}"]
            params.append((np.array(layer["kernel"]).T,np.array(layer["bias"])))
        observation_dim=params[0][0].shape[0]
        net=cls(observation_dim,model.actor.action_dim,model.actor.net_arch)
        with torch.no_grad():
            for l,p in zip([net for net in net.common if isinstance(net,nn.Linear)],params):
                l.weight=nn.Parameter(torch.from_numpy(p[0]))
                l.bias=nn.Parameter(torch.from_numpy(p[1]))
            net.mu.weight=nn.Parameter(torch.from_numpy(params[-1][0]))
            # print(net.mu.weight.shape,params[-1][0].shape)
            net.mu.bias=nn.Parameter(torch.from_numpy(params[-1][1]))
        return net
    
def onnx_export(model, env, onnx_path: str = "sk8o_actor.onnx"):
    if isinstance(model,DroQ):
        onnxable_model=OnnxableDroQJAX.from_sbx(model)
        observation_shape=(20,6)
        dummy_input=torch.randn(observation_shape).to("cpu")
        inputs = "inputs"
        outputs = "outputs"
    else:
        onnxable_model = OnnxablePolicy(model.policy.actor)
        observation_size = model.observation_space.shape
        # TODO: what is the dummy input for?
        dummy_input = torch.randn(observation_size).to("cpu")
        try:
            if isinstance(env, DummyVecEnv):
                env = env.envs[0]
            inputs = f"{observation_size[0]}x[" + "-".join(env.observation_names) + "]"
            outputs = "-".join(env.action_names)
        except AttributeError:
            inputs = "inputs"
            outputs = "outputs"
    torch.onnx.export(
        onnxable_model.to("cpu"),
        dummy_input,
        onnx_path,
        # opset_version=9,
        input_names=[inputs],
        output_names=[outputs],
    )

    # ort_sess = ort.InferenceSession(onnx_path)
    # return ort_sess
