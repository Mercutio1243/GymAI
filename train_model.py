import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from torchviz import make_dot
import torch

import matrix_env


policy_kwargs = dict(net_arch=[64])
n=8
num_zeros=1

envTrain = gym.make('MatrixEnv-v0', n=n, num_zeros=num_zeros)
envTrain = Monitor(envTrain)

envTest = gym.make('MatrixEnv-v0', n=n, num_zeros=num_zeros)

model = DQN('MlpPolicy', envTrain, verbose=1, policy_kwargs=policy_kwargs)

model.learn(total_timesteps=10000, log_interval=10000)

#eval_env = gym.make('MatrixEnv-v0')
#eval_env = Monitor(eval_env)

#mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
#print(f"Mean reward: {mean_reward} +/- {std_reward}")

#print(model.policy)

# Test the trained model
obs, info = envTest.reset()
for step in range(20):
    envTest.renderstatex()
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = envTest.step(action)
    envTest.renderstepx(obs, action, reward, done, step)

    if done:
        print("Episode finished")
        obs, info = envTest.reset()
        print(f"+++++")

envTrain.close()
envTest.close()
