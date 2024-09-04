import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "Acrobot-v1"                         # specify the environment to use
log_path = os.path.join('Training', 'Logs')     # make Directories

env = gym.make(env_name)                        # create the gym environment and wrap it in a DummyVecEnv for vectorization
env = DummyVecEnv([lambda: env])                #

hyperparams = {
    "learning_rate": 0.001,
    "gamma": 0.98,
	"batch_size": 96,
	"n_epochs": 8,
    "clip_range": 0.3,
}

# create the PPO model with a multi-layer perceptron policy, and set up logging
model = PPO('MlpPolicy', env, verbose=1, **hyperparams, tensorboard_log=log_path)
# train the model for a total of 100000 time steps
model.learn(total_timesteps=100000)


PPO_Path = os.path.join('Training', 'Saved Models', 'PPO_Model_Acrobot_Params')   # specify the path to save the trained model
model.save(PPO_Path)                                                        # save the trained model
del model                                                                   # free up memory by deleting the untrained model
model = PPO.load(PPO_Path, env=env)                                         # load the saved model
