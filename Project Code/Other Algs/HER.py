import os
import gym
import matplotlib.pyplot as plt 
from stable_baselines3 import HER
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "Acrobot-v1"

log_path = os.path.join('Training', 'Logs')  #Make Directories

env = gym.make(env_name)
env = DummyVecEnv([lambda: env])
model = HER('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=30000)


PPO_Path = os.path.join('Training', 'Saved Models', 'Her_Model_Acrobot')
model.save(PPO_Path)
del model
model = HER.load(PPO_Path, env=env)
