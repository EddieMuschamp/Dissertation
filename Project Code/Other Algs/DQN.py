import os
import gym
import matplotlib.pyplot as plt 
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "CartPole-v1"

log_path = os.path.join('Training', 'Logs')  #Make Directories

env = gym.make(env_name, render_mode='human')
env = DummyVecEnv([lambda: env])
model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=10000, log_interval=4)
DQN_Path = os.path.join('Training', 'Saved Models', 'DQN_Model_Cartpole')
model.save(DQN_Path)
del model
model = DQN.load(DQN_Path, env=env)

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
