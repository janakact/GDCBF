import gymnasium as gym
import numpy as np
from gymnasium.wrappers.flatten_observation import FlattenObservation
from jaxrl5.wrappers.single_precision import SinglePrecision
# import gym
# from gym.wrappers.flatten_observation import FlattenObservation
# from jaxrl5.wrappers.single_precision import SinglePrecision


class OfflineEnvWrapper(gym.Wrapper):
    """
    Wrapper class for offline RL envs.
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.noise_scale = None

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        if self.noise_scale is not None:
            obs += np.random.normal(0, self.noise_scale, obs.shape)
        return obs, info

    def set_noise_scale(self, noise_scale):
        self.noise_scale = noise_scale

    def step(self, action):
        obs_next, reward, terminated, truncated, info = self.env.step(action)
        if self.noise_scale is not None:
            obs_next += np.random.normal(0, self.noise_scale, obs_next.shape)
        return obs_next, reward, terminated, truncated, info

def wrap_gym(env: gym.Env, rescale_actions: bool = True, cost_limit: int = 1) -> gym.Env:
    env = SinglePrecision(env)

    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1, 1)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env = gym.wrappers.ClipAction(env)
    env.set_target_cost(cost_limit)
    print('env_cost_limit', env.target_cost)
    return env
