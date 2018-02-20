"""ADJUSTED FOR TASK 1"""

import numpy as np
from quad_controller_rl.agents.base_agent import BaseAgent
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers import Input, Dense
import random
from collections import namedtuple

Experience_Replay = namedtuple("Experience_replay",
                               field_names=['state', 'action', 'reward',
                                            'next_state', 'done'])


class Actor:
  def __init__(self, state_size, action_size, action_low, action_high):
      self.state_size = state_size
      self.action_size = action_size
      self.action_low = action_low
      self.action_high = action_high
      self.action_range = self.action_high = self.action_low
      # build the NN model
      self.build_NN()
  def build_NN(self):
      states = layers.Input(shape=(self.state_size,), name='states')
      # add dense layers
      x = layers.Dense(units=16, activation='relu')(states)
      x = layers.Dense(units=32, activation='relu')(x)
      x = layers.Dense(units=64, activation='relu')(x)
      # squish outputs between 0 and 1
      raw_actions = layers.Dense(units=self.action_size,
                                 activation='sigmoid',
                                 name='raw_actions')(x)
      # re scale the values to appropriate range
      actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
                              name='actions')(raw_actions)
      # create the model
      self.model = models.Model(inputs=states, outputs=actions)
      # define loss function
      action_gradients = layers.Input(shape=(self.action_size,))
      loss = K.mean(-action_gradients*actions)
      # define optimizer
      optimizer = optimizers.Adam()
      # define training function
      updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
      self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()],
                                 outputs=[], updates=updates_op)
      
      
Class Critic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # build the NN model
        self.build_NN()
        
    def build_NN(self):
        # initialize state and action inputs
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')
        # layers for state network
        net_states = layers.Dense(units=32, activation='relu')(states)
        net_states = layers.Dense(units=64, activation='relu')(net_states)
        # layers for actions network
        net_actions = layers.Dense(units=32, activation='relu')(actions)
        net_actions = layers.Dense(units=64, activation='relu')(net_actions)
        # combine state and actions
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)
        # generate a Q_value
        Q_values = layers.Dense(units=1, name='q_values')(net)
        # create the model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)
        # set the optimizer
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')
        # calculate the gradients with respect to actions
        action_gradients = K.gradients(Q_values, actions)
        # function to get the action gradients
        # to be used by Actor
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)


class ReplayBuffer:
    def __init__(self, size=100):
        self.size = size
        self.memory = []
        self.idx = 0
    def add(self, state, action, reward, next_state, done):
        set = Experience_replay(state, action, reward, next_state, done)
        if len(self.memory) < self.size:
            self.memory.append(set)
        else:
            self.memory[self.idx] = set
            self.idx = (self.idx + 1) % self.size
    def sample(self, batch_size=16):
        return random.sample(self.memory, batch_size)
    def reset(self):
        self.memory[:] = []
    def __len__(self):
        return len(self.memory)

class Task1_Policy(BaseAgent):

    def __init__(self, task):
        # Task (environment) information
        self.task = task  # should contain observation_space and action_space
        self.state_size = np.prod(self.task.observation_space.shape)
        self.state_range = self.task.observation_space.high - self.task.observation_space.low
        self.action_size = np.prod(self.task.action_space.shape)
        self.action_range = self.task.action_space.high - self.task.action_space.low

        # Policy parameters
        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size)).reshape(1, -1))  # start producing actions in a decent range
        
        # Nueral network as value function
        self.deep_Q = self.build_NN(nuerons=128, activation='relu', optimizer='adam')
        self.experience_replay = []
        self.batch_size = 16
        self.gamma = 0.9
        
        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        # Episode variables
        self.reset_episode_vars()

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0
        # reset the experience replay buffer
        self.experience_replay[:] = []
        
    def step(self, state, reward, done):
        # Transform state vector
        # print ("reward:\t", reward)
        state = (state - self.task.observation_space.low) / self.state_range  # scale to [0.0, 1.0]
        state = state.reshape(1, -1)  # convert to row vector
        
        # Choose an action
        action = self.act(state)
        
        self.experience_replay.append((self.last_state, self.last_action, reward, state, done))
        
        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.total_reward += reward
            self.count += 1

        # Learn, if at end of episode
        if done:
            self.learn()
            self.reset_episode_vars()

        self.last_state = state
        self.last_action = action
        return action

    def act(self, state):
        # Choose action based on given state and policy
        # action = np.dot(state, self.w)  # simple linear policy
        
        print ("Input state:\t{0}".format(state))
        
        action = self.deep_Q.predict(state)
        
        print ("Output action:\t{0}".format(action))
        
        return action[0]

    def learn(self):
        
        print ("learning")
        batch_size = len(self.experience_replay)//2
        random_sample = random.sample(self.experience_replay, batch_size)
        
        for state, action, reward, next_state, done in random_sample:
            print ("\nNew Iteration...")
            print ("state:\t", state)
            print ("action:\t", action)
            print ("reward:\t", reward)
            print ("next_state:\t", next_state)
            print ("done?\t", done)
            
            target = reward
            
            if not done:
                target = reward + self.gamma * np.amax(self.deep_Q.predict(next_state)[0])
                
            target_f = self.deep_Q.predict(state)
            target_f[0][action] = target
            
            self.deep_Q.fit(state, target_f, epochs=1, verbose=0)
            
        print ("Done training!")
        
        
    # method for building a simple nueral network
    def build_NN(self, nuerons=64, activation='relu', optimizer='adam'):
        model = Sequential()
        model.add(Dense(units=nuerons,
                        activation = activation,
                        input_dim=self.state_size))
        model.add(Dropout(.2))
        model.add(Dense(units=nuerons//2,
                        activation = activation))
        model.add(Dropout(.2))
        model.add(Dense(units=self.action_size,
                        activation = 'linear'))
        model.compile(loss = 'mse', optimizer=optimizer) # , metrics=['accuracy'])
        model.summary()
        return model
