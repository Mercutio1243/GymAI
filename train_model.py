import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from torchviz import make_dot
import torch

import matrix_env


size_space=5
size_piece=2
num_zeros=5
policy_kwargs = dict(net_arch=[64])

envTrain = gym.make('MatrixEnv-v0', size_space=size_space, size_piece=size_piece, num_zeros=num_zeros)
envTrain = Monitor(envTrain)

envTest = gym.make('MatrixEnv-v0', size_space=size_space, size_piece=size_piece, num_zeros=num_zeros)

model = DQN('MlpPolicy', envTrain, verbose=2, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=1000000, log_interval=10000)



envEval = gym.make('MatrixEnv-v0', size_space=size_space, size_piece=size_piece, num_zeros=num_zeros)
envEval = Monitor(envEval)

mean_reward, std_reward = evaluate_policy(model, envEval, n_eval_episodes=1000)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

print(model.policy)

# Test the trained model
obs, info = envTest.reset()
for step in range(20):
    envTest.unwrapped.renderstate() 
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = envTest.step(action)
    obs_matrix_space_after_insertion = info['combined_obs_after_insertion'].flatten()
    envTest.unwrapped.renderstep(obs_matrix_space_after_insertion, action, reward, done, step)  
    
    if done:
        print("Episode finished")
        obs, info = envTest.reset()
        print(f"+++++")

envTrain.close()
envTest.close()
