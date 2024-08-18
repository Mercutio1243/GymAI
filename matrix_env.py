import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MatrixEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, size_space=5, size_piece=2, num_zeros=1):
        if size_piece >= size_space:
            raise ValueError(f"size_piece should be less than size_space. Received size_piece={size_piece} and size_space={size_space}.")
        super(MatrixEnv, self).__init__()
        self.size_space= size_space
        self.size_piece = size_piece
        self.num_zeros = num_zeros

        self.observation_space = spaces.MultiBinary(size_space * size_space + size_piece * size_piece)
        self.action_space = spaces.Discrete((size_space-size_piece+1)*(size_space-size_piece+1))

        self.state_matrix_space = self._initialize_matrix_space()
        self.state_matrix_piece = self._initialize_matrix_piece()
        self._make_room_for_piece_in_space()

    def _initialize_matrix_piece(self):
        # Initialize the piece matrix and ensure it has at least one True value
        while True:
            state = np.random.choice([True, False], size=(self.size_piece, self.size_piece))
            if np.any(state):  # Check if there's at least one True value
                break  # Exit the loop if the condition is met
        return state

    def _initialize_matrix_space(self):
        # Start by initializing the matrix with the given number of True values
        state = np.zeros((self.size_space, self.size_space), dtype=bool)
        true_indices = np.random.choice(self.size_space * self.size_space, self.size_space * self.size_space - self.num_zeros, replace=False)
        state.flat[true_indices] = True
        self.state_matrix_space = state
        
        return self.state_matrix_space

    def _make_room_for_piece_in_space(self):
        # Randomly choose a position where the piece will be placed
        row = np.random.randint(0, self.size_space - self.size_piece + 1)
        col = np.random.randint(0, self.size_space - self.size_piece + 1)
        
        # Extract the submatrix where the piece will be placed
        space_submatrix = self.state_matrix_space[row:row + self.size_piece, col:col + self.size_piece]
        
        # Clear the area in the space matrix where the piece matrix has True values
        space_submatrix[self.state_matrix_piece] = False


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state_matrix_space = self._initialize_matrix_space()
        self.state_matrix_piece = self._initialize_matrix_piece()
        self._make_room_for_piece_in_space()

        # Combine both matrices into a single flattened observation
        combined_obs = np.concatenate([self.state_matrix_space.flatten(), self.state_matrix_piece.flatten()])
        return combined_obs, {}

    def step(self, action):
        matrix_space_after_insertion = self.state_matrix_space.copy()
        matrix_piece_after_insertion = self.state_matrix_piece.copy()
        # Calculate the top-left corner of where the piece will be placed in the space matrix
        row = action // (self.size_space - self.size_piece + 1)
        col = action % (self.size_space - self.size_piece + 1)

        done = False
        reward = 0

        # Check if the space is available to place the piece
        space_submatrix = self.state_matrix_space[row:row+self.size_piece, col:col+self.size_piece]

        if np.any(space_submatrix & self.state_matrix_piece):  # If there's overlap with True values
            reward = -1
            done = True  # End the episode if an invalid move is made
        else:
            # Place the piece in the space matrix
            self.state_matrix_space[row:row+self.size_piece, col:col+self.size_piece] |= self.state_matrix_piece
            reward = 1

            # Save space matrix after insertion
            matrix_space_after_insertion = self.state_matrix_space.copy()

            # After successfully placing the piece, initialize a new piece and ensure space for it
            self.state_matrix_piece = self._initialize_matrix_piece()
            self._make_room_for_piece_in_space()

        # Check if all cells in state_matrix_space are now True
    #    if np.all(self.state_matrix_space):
    #        reward = 2
    #        done = True

        truncated = False

        combined_obs_after_insertion = np.concatenate([matrix_space_after_insertion.flatten(), matrix_piece_after_insertion.flatten()])
        info = {
            'combined_obs_after_insertion': combined_obs_after_insertion
        }

        combined_obs = np.concatenate([self.state_matrix_space.flatten(), self.state_matrix_piece.flatten()])

        return combined_obs, reward, done, truncated, info

    def renderstate(self, mode='human'):
        print(f"Space:\n{self.state_matrix_space.astype(int)}")
        print(f"Piece:\n{self.state_matrix_piece.astype(int)}")

    def renderstep(self, obs, action, reward, done, step, mode='human'):
        # Reshape the observation back into the two matrices
        obs_matrix_state = obs[:self.size_space * self.size_space].reshape(self.size_space, self.size_space).astype(int)
        obs_matrix_piece = obs[self.size_space * self.size_space:].reshape(self.size_piece, self.size_piece).astype(int)
        
        # Calculate the top-left corner of where the piece will be placed
        row = action // (self.size_space - self.size_piece + 1)
        col = action % (self.size_space - self.size_piece + 1)
        
        # Create an action matrix to visualize the placement of the piece
        action_matrix = np.zeros((self.size_space, self.size_space), dtype=int)
        action_matrix[row:row + self.size_piece, col:col + self.size_piece] = obs_matrix_piece
        
        # Print the step details, including the updated action visualization
        print(f"Step: {step}, \nAction (Piece Placement): \n{action_matrix}, \nSpace: \n{obs_matrix_state}, \nReward: {reward}")
        print("#####")

    def close(self):
        pass

from gymnasium.envs.registration import register

# Register the environment
register(
    id='MatrixEnv-v0',
    entry_point='matrix_env:MatrixEnv',
)
