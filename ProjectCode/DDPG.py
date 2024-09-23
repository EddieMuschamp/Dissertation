import os
import gym
import matplotlib.pyplot as plt 
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "Acrobot-v1"

log_path = os.path.join('Training', 'Logs')  #Make Directories

env = gym.make(env_name)
env = DummyVecEnv([lambda: env])
model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=30000)


DQN_Path = os.path.join('Training', 'Saved Models', 'DDPG_Model_Acrobot')
model.save(DQN_Path)
del model
model = DDPG.load(DQN_Path, env=env)
