import gym
import random
import numpy as np
from keras import Sequential
from keras import optimizers
from keras.layers import Dense
from collections import deque

## Hyper Parameters:
episodes = 10000
time = 1000
# gamma: discount rate
gamma = 0.95
batch_size = 32
# learning rate: determine how much neural net learns in each iteration
learning_rate = 0.001
memory_len = 10000

# use epsilon-greedy to deal with trade-off between exploration and exploitation
# epsilon_min: guarantee the least level of exploration
epsilon_min = 0.01
# epsilon_decay: decay rate
epsilon_decay = 0.995

class Agent:
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    
    # initialize epsilon: exploration rate
    self.epsilon = 1.0
    # use deque to dynamically collect experiences in memory
    self.memory = deque( maxlen = memory_len)
    
    self.model = self.build_model()

  def build_model(self):
    
    # create foundations of layers
    model = Sequential()

    # input layer with dimension of state size
    model.add( Dense(24, input_dim = self.state_size, activation = 'relu') )

    # hidden layer
    model.add( Dense(24, activation = 'relu') )

    # output layer
    model.add( Dense(self.action_size, activation = 'linear') )

    # create the model
    model.compile( loss = 'mse', optimizer = optimizers.RMSprop(lr = learning_rate) )
    
    return model
  
  
  def remember(self, state, action, reward, next_state, done):
    # collect the previous experiences in memory
    self.memory.append( (state, action, reward, next_state, done) )
  
  def act(self, state):
    # use epsilon-greedy to deal with trade-off between exploration and exploitation
    if random.random() <= self.epsilon:
      # exploration: acts randomly
      return env.action_space.sample()
    else:
      # exploitation
      return np.argmax( self.model.predict(state) )
    
  def replay(self):
    # experience replay: randomly sample from memory
    mini_batch = random.sample( self.memory, batch_size )

    # deal with each experience from minibatch
    for state, action, reward, next_state, done in mini_batch:
      #print(state.shape)
      if done:
        # approach the termication of this eposide
        target = reward
      else:
        target = reward + gamma * np.max( self.model.predict(next_state)[0] )
      
      # predict the reward of current state
      prediction = self.model.predict(state)

      # update the q value of specific action of current state
      prediction[0][action] = target
      
      self.model.fit(state, prediction, epochs = 1, verbose = 0)

if __name__ == "__main__":
  
  # initialize gym environment
  env = gym.make('CartPole-v0')
  
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n
  #initialize agent
  agent = Agent(state_size, action_size)
  

  for e in range(episodes):
    state = env.reset()
    state = np.reshape( state, [1,state_size] )
    
    for t in range(time):
      #env.render()
      action = agent.act(state)
      next_state, reward, done, info = env.step(action)
      next_state = np.reshape( next_state, [1, state_size] )
      reward = 1 if not done else -10
      agent.remember(state, action, reward, next_state, done)
      state = next_state
      
      if done:
        print( "episode: {}/{}, score: {}, e: {:.2}".format(e, episodes, t, agent.epsilon) )
        break
      
      if len(agent.memory) > batch_size:
        agent.replay()
        
    # epsilon decay
    agent.epsilon = max(agent.epsilon * epsilon_decay, epsilon_min)
