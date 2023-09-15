# start code: https://github.com/pairlab/d2rl/blob/main/sac/model.py

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.distributions import (
    SquashedDiagGaussianDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import Actor, SACPolicy
from torch import nn
from torch.distributions import Normal

# CAP the standard deviation of the actor
LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
epsilon = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class D2RLCritic(ContinuousCritic):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(ContinuousCritic, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = D2RLQNetwork(features_dim, action_dim, net_arch)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)


class D2RLQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, net_arch):
        super(D2RLQNetwork, self).__init__()
        self.layers = [nn.Linear(num_inputs + num_actions, net_arch[0])]
        last_dim = net_arch[0]
        for n in net_arch[1:]:
            in_dim = num_inputs + num_actions + last_dim
            self.layers.append(nn.Linear(in_dim, n))
            last_dim = n
        self.out = nn.Linear(net_arch[-1], 1)
        self.layers = nn.ModuleList(self.layers)
        self.apply(weights_init_)

    def forward(self, x_in):
        x = x_in
        for l, layer in enumerate(self.layers):
            if l > 0:
                x = torch.cat([x, x_in], dim=-1)
            x = F.relu(layer(x))
        x = self.out(x)
        return x


class D2RLActor(Actor):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super(Actor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        action_dim = get_action_dim(self.action_space)
        self.latent_pi = D2RLGaussianPolicy(features_dim, net_arch)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim,
                full_std=full_std,
                use_expln=use_expln,
                learn_features=True,
                squash_output=True,
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim,
                latent_sde_dim=last_layer_dim,
                log_std_init=log_std_init,
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(
                    self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean)
                )
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)  # type: ignore[assignment]
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)  # type: ignore[assignment]


class D2RLGaussianPolicy(nn.Module):
    def __init__(self, num_inputs, net_arch):
        super(D2RLGaussianPolicy, self).__init__()
        self.layers = [nn.Linear(num_inputs, net_arch[0])]
        last_dim = net_arch[0]
        for n in net_arch[1:]:
            in_dim = num_inputs + last_dim
            self.layers.append(nn.Linear(in_dim, n))
            last_dim = n
        self.layers = nn.ModuleList(self.layers)
        self.apply(weights_init_)

    def forward(self, x_in):
        x = x_in
        for l, layer in enumerate(self.layers):
            if l > 0:
                x = torch.cat([x, x_in], dim=-1)
            x = F.relu(layer(x))
        return x

    # def sample(self, state):
    #     mean, log_std = self.forward(state)
    #     std = log_std.exp()
    #     normal = Normal(mean, std)
    #     x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
    #     y_t = torch.tanh(x_t)
    #     action = y_t * self.action_scale + self.action_bias
    #     log_prob = normal.log_prob(x_t)
    #     # Enforcing Action Bound
    #     log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
    #     log_prob = log_prob.sum(1, keepdim=True)
    #     mean = torch.tanh(mean) * self.action_scale + self.action_bias
    #     return (
    #         action,
    #         log_prob,
    #     )


class D2RL_SACPolicy(SACPolicy):
    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Actor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return D2RLActor(**actor_kwargs).to(self.device)

    def make_critic(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> D2RLCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return D2RLCritic(**critic_kwargs).to(self.device)
