import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MatrixEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, n=5, num_zeros=1):
        super(MatrixEnv, self).__init__()
        self.n = n
        self.num_zeros = num_zeros
        self.action_space = spaces.Discrete(n * n)  # Single integer for flattened grid
        self.observation_space = spaces.Box(low=0, high=1, shape=(n, n), dtype=np.int32)
        self.state = self._initialize_state()

    def step(self, action):
        # Map the action back to row, col
        row = action // self.n
        col = action % self.n

        done = False

        if self.state[row, col] == 0:
            self.state[row, col] = 1
            reward = 1
        else:
            reward = -10
            done = True

        if np.all(self.state == np.ones((self.n, self.n), dtype=np.int32)):
            reward = 10
            done = True

        truncated = False

        return self.state, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self._initialize_state()
        return self.state, {}

    def _initialize_state(self):
        num_zeros = min(self.num_zeros, self.n * self.n)
        num_ones = self.n * self.n - num_zeros

        elements = np.array([1] * num_ones + [0] * num_zeros)
        np.random.shuffle(elements)
        state = elements.reshape((self.n, self.n))

        return state

    def render(self, mode='human'):
        print(f"State:\n{self.state}")

    def close(self):
        pass

from gymnasium.envs.registration import register

# Register the environment
register(
    id='MatrixEnv-v0',
    entry_point='matrix_env:MatrixEnv',
)
