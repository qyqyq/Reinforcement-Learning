import gym
import numpy as np
import random

# Build the environment
env = gym.make("FrozenLake-v0")

# initialize the Q Table
action = env.action_space.n
state = env.observation_space.n
qtable = np.zeros((state, action))

# Set the hyperparameters
num_episode = 1000          # total number of episodes
alpha = 0.8                 # learning rate
num_step = 100              # total number of steps per episode
gamma = 0.95                # discount rate
# Tradeoff between exploration and exploitation
epsilon = 1.0               # the extent of greed
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005          # exponential decay rate

# Q Learning Algorithm
rewards = []

for episode in range(num_episode):
  
  # initialization for each episode
  state = env.reset()
  done = False              # indicate whether this episode is over
  sum_reward = 0
  
  for step in range(num_step):
    tradeoff = random.random()
    
    if tradeoff < epsilon:  # exploration
      action = env.action_space.sample()
    else:
      action = np.argmax(qtable[state, :])
      
    # next step:
    new_state, reward, done, info = env.step(action)
    qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])    
    
    sum_reward += reward
    if done == True:
      break
    state = new_state
      
  rewards.append(sum_reward)
  
  #decay epsilon, induce less and less exploration
  epsilon = min_epsilon + (max_epsilon-min_epsilon)*np.exp(-decay_rate*epsilon)
  
print(qtable)

# use "cheat sheet" to play
env.reset()

for episode in range(15):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("Episode: ", episode)

    for step in range(num_step):
        
        action = np.argmax(qtable[state,:])
        new_state, reward, done, info = env.step(action)
        
        if done:
            env.render()    # show the status of agent and environment
            print("Number of steps", step)
            break
            
        state = new_state
env.close()
