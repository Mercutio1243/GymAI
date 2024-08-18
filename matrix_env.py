import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MatrixEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, n=5, x=2, num_zeros=1):
        if x >= n:
            raise ValueError(f"x should be less than n. Received x={x} and n={n}.")
        super(MatrixEnv, self).__init__()
        self.n = n
        self.x = x
        self.num_zeros = num_zeros
        self.action_space = spaces.Discrete(n * n)
        self.observation_space = spaces.MultiBinary(n * n)  # Binary observation space
        self.state = self._initialize_state()

    def _initialize_state(self):
        # Generate a boolean array where `False` represents zero and `True` represents one
        state = np.zeros((self.n, self.n), dtype=bool)
        # Randomly set a specified number of `True` values in the state
        true_indices = np.random.choice(self.n * self.n, self.n * self.n - self.num_zeros, replace=False)
        state.flat[true_indices] = True
        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self._initialize_state()
        return self.state.flatten(), {}  # Return a flattened version of the state

    def step(self, action):
        # Convert the action (which is a single integer) back to row and column indices
        row = action // self.n
        col = action % self.n

        done = False
        reward = 0

        # Determine reward based on the action
        if self.state[row, col] == False:  # If the selected cell is False (0)
            self.state[row, col] = True    # Set it to True (1)
            reward = 1
        else:  # If the selected cell is already True (1)
            reward = -10
            done = True  # End the episode if an invalid move is made

        # Check if all cells are now True
        if np.all(self.state):
            reward = 10
            done = True

        truncated = False

        return self.state.flatten(), reward, done, truncated, {}


    def renderstatex(self, mode='human'):
        print(f"State:\n{self.state.astype(int)}")

    def renderstepx(self, obs, action, reward, done, step, mode='human'):
        obs_matrix = obs.reshape(self.n, self.n).astype(int)
        row = action // self.n
        col = action % self.n
        action_matrix = np.zeros((self.n, self.n), dtype=int)
        action_matrix[row, col] = 1
        print(f"Step: {step}, \nAction: \n{action_matrix}, \nState: \n{obs_matrix}, \nReward: {reward}")
        print(f"#####")

    def close(self):
        pass

from gymnasium.envs.registration import register

# Register the environment
register(
    id='MatrixEnv-v0',
    entry_point='matrix_env:MatrixEnv',
)