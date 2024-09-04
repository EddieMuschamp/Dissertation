import os
import gym
import matplotlib.pyplot as plt 
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "CartPole-v1"

log_path = os.path.join('Training', 'Logs')  #Make Directories

env = gym.make(env_name)
env = DummyVecEnv([lambda: env])
model = TD3('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=10000, log_interval=4)


DQN_Path = os.path.join('Training', 'Saved Models', 'TD3_Model_Cartpole')
model.save(DQN_Path)
del model
model = TD3.load(DQN_Path, env=env)
