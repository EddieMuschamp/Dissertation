import os
import gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "Acrobot-v1"
env = gym.make(env_name, render_mode='human')

model_Path = os.path.join('Training', 'Saved Models', 'PPO_Model_Acrobot_Params')
model = PPO.load(model_Path, env=env)

print(evaluate_policy(model, env, n_eval_episodes=10, render=True))

env.close()