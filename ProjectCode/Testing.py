import os
import gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from sb3_contrib import TRPO
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "Acrobot-v1"                                                             # specify the environment to use
env = gym.make(env_name, render_mode='human')                                       # create the gym environment
model_Path = os.path.join('Training', 'Saved Models', 'A2C_Model_Acrobot_Params')  # retrieve the trained model
model = TRPO.load(model_Path, env=env)                                              # and load it
episodes = 100                                                                      # number of episodes the agent runs for 
for episode in range(1, episodes+1):                                                # run algorithm in environment for each episode
    obs, _ = env.reset()                                                            # set the observation as the start of the environment
    terminated = False                                                              #initialize episode-specific variables
    score = 0                                                                       #
    while not terminated:                                                           # loop while not terminated
        env.render()                                                                # render the environment
        action, _ = model.predict(obs)                                              # action is predicted by algorithm
        obs, reward, terminated, truncated, info = env.step(action)                 # take the action
        score += reward                                                             # calculate reward
    print('Episode:{} Score: {}'.format(episode, score))                            # output score for each episode

env.close()                                                                         # close the environment

