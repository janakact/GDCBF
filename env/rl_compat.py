import os

# 1. Detect Library
LIB_MODE = os.getenv("RL_LIB", "gymnasium")  # Defaults to gymnasium

if LIB_MODE == "gymnasium":
    print("Using gymnasium")
    import gymnasium as gym
    from gymnasium.wrappers.flatten_observation import FlattenObservation

else:
    print("Using gym")
    import gym
    from gym.wrappers.flatten_observation import FlattenObservation

# Re-export the gym module references you need
spaces = gym.spaces
