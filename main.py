import gym
import numpy as np
import random


from agent import PGAgent

# Global variables
GAMMA = 0.99

# Get Environment
env = gym.make("InvertedPendulum-v1")
print env.observation_space

# Get Env Variables
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
low = env.action_space.low[0]
high = env.action_space.high[0]

# Global Variables
NUM_EPISODES = 1000
GAMMA = 0.99

# Creat Agent object
bot = PGAgent(state_dim, action_dim, GAMMA, low, high)

for i in range(NUM_EPISODES):
    transitions = [] # Consists of [state, action, reward, n_state, terminal]

    finish = 0

    obs = env.reset()

    obs = np.reshape(obs, (1, state_dim))

    totalReward = 0
    while finish == 0:

        env.render()

        action = bot.select_action(obs)

        #print "Current Obs : ", obs

        n_obs, reward, done, _ = env.step(action)

        n_obs = np.reshape(n_obs, (1, state_dim))

        #print "Next State : ", n_obs

        if done :
            done = 1

        else:
            done = 0

        transitions.append([obs, action, reward, n_obs, done])

        #print "Transitions : ", transitions

        totalReward += reward

        obs = n_obs

        if done:

            finish = 1

            print "Episode : ", i, " Reward : ", totalReward

            bot.finish_episode(transitions)









