import gym
env_name = "Acrobot-v1"                         # specify the environment to use
env = gym.make(env_name, render_mode='human')   # create an instance of the environment
episodes = 100                                  # Number of episodes the agent runs for 

for episode in range(1, episodes + 1):          # loop over each episode
    state = env.reset()                         # reset the environment to the initial state

    terminated = False                          # initialize episode-specific variables
    score = 0                                   #
    
    while not terminated:                       # loop until the episode is terminated
        env.render()                            # render the environment
        
        action = env.action_space.sample()      # choose a random action

        obs, reward, terminated, truncated, info = env.step(action) # take the action

        score += reward                         # update the episode score
    
    print('Episode:{} Score: {}'.format(episode, score))            # print the episode score

env.close()                                     # Close the env