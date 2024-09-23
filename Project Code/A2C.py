import os
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "Acrobot-v1"

log_path = os.path.join('Training', 'Logs')  #Make Directories

#Hyperparameters:
hyperparams = {
    "learning_rate": 0.001, 
    "gamma": 0.98,
}



env = gym.make(env_name)
env = DummyVecEnv([lambda: env])
model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=100000)


model_Path = os.path.join('Training', 'Saved Models', 'A2C_Model_Acrobot')
model.save(model_Path)
del model
model = A2C.load(model_Path, env=env)
