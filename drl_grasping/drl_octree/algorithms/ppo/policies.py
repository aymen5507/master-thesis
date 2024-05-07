from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import ocnn
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy, register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import is_vectorized_observation
from stable_baselines3.ppo.policies import MlpPolicy
from torch import nn

from drl_grasping.drl_octree.features_extractor import (
    ImageCnnFeaturesExtractor,
    OctreeCnnFeaturesExtractor,
)
from drl_grasping.drl_octree.replay_buffer import (
    preprocess_stacked_depth_image_batch,
    preprocess_stacked_octree_batch,
)


class OctreeCnnPolicyPPO(MlpPolicy):
    """
    Policy class (with both actor and critic) for PPO.
    Overridden to not preprocess observations (unnecessary conversion into float)
    and to handle octree observations.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the feature extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: Type[BaseFeaturesExtractor] = OctreeCnnFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        separate_networks_for_stacks: bool = True,
        debug_write_octree: bool = False,
    ):
        features_extractor_kwargs.update({"separate_networks_for_stacks": separate_networks_for_stacks})
        super(OctreeCnnPolicyPPO, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

        self._separate_networks_for_stacks = separate_networks_for_stacks
        self._debug_write_octree = debug_write_octree

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.
        Overridden to skip pre-processing (for some reason it converts tensor to Float)

        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No features extractor was set"
        return self.features_extractor(obs)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Overridden to create proper Octree batch.
        Get the policy action and state from an observation (and optional state).

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not isinstance(observation, dict):
            observation = np.array(observation)

        vectorized_env = is_vectorized_observation(observation, self.observation_space)

        if self._debug_write_octree:
            ocnn.write_octree(th.from_numpy(observation[-1]), "octree.octree")

        # Make batch out of tensor (consisting of n-stacked octrees)
        octree_batch = preprocess_stacked_octree_batch(
            observation,
            self.device,
            separate_batches=self._separate_networks_for_stacks,
        )

        with th.no_grad():
            actions, values, log_probs = self._predict(octree_batch, deterministic=deterministic)
        # Convert to numpy
        actions = actions.cpu().numpy()
        values = values.cpu().numpy()
        log_probs = log_probs.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions = actions[0]

        return actions, state


class DepthImageCnnPolicyPPO(MlpPolicy):
    """
    Policy class (with both actor and critic) for PPO.
    Overridden to not preprocess observations (unnecessary conversion into float)
    and to handle depth image observations.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the feature extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: Type[BaseFeaturesExtractor] = ImageCnnFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        separate_networks_for_stacks: bool = True,
    ):
        features_extractor_kwargs.update({"separate_networks_for_stacks": separate_networks_for_stacks})
        super(DepthImageCnnPolicyPPO, self).__