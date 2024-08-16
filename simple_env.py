import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimpleEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self):
        super(SimpleEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # Three possible actions
        self.observation_space = spaces.Discrete(5)  # Five possible states
        self.state = 0

    def step(self, action):
        if action == 0:
            self.state = max(0, self.state - 1)
        elif action == 1:
            self.state = min(4, self.state + 1)
        elif action == 2:
            self.state = 0

        reward = 0
        if self.state == 4:
            reward = 10
        else:
            reward = -0.1

        done = self.state == 4
        truncated = False  # Since we don't have a truncation condition

        return self.state, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return self.state, {}

    def render(self, mode='human'):
        print(f"State: {self.state}")

    def close(self):
        pass

from gymnasium.envs.registration import register

# Register the environment
register(
    id='SimpleEnv-v0',
    entry_point='simple_env:SimpleEnv',
)