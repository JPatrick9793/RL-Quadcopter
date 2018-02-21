"""ADJUSTED FOR TASK 1"""

import numpy as np
from quad_controller_rl.agents.base_agent import BaseAgent
import keras
from keras import layers, models, optimizers
from keras import backend as K
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
      
      
class Critic:
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
    def __init__(self, size=10):
        self.size = size
        self.memory = []
        self.idx = 0
    def add(self, state, action, reward, next_state, done):
        set = Experience_Replay(state, action, reward, next_state, done)
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
        self.task = task
        # self.state_size = np.prod(self.task.observation_space.shape)
        self.state_size = 3
        self.state_range = self.task.observation_space.high - self.task.observation_space.low
        
        # self.action_size = np.prod(self.task.action_space.shape)
        self.action_size = 3
        # self.action_low = task.action_space.low
        self.action_low = task.action_space.low[0:3]
        # self.action_high = task.action_space.high
        self.action_high = task.action_space.high[0:3]
        self.action_range = self.action_high - self.action_low
        print("Original spaces: {}, {}\nConstrained spaces: {}, {}".format(
            self.task.observation_space.shape, self.task.action_space.shape,
            self.state_size, self.action_size))
        
        # Actor object
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        # duplicate Actor object for fixed Q
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        
        # Critic object
        self.critic_local = Critic(self.state_size, self.action_size)
        # duplicate Critic object for fixed Q
        self.critic_target = Critic(self.state_size, self.action_size)
        
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        
        self.noise = OUNoise(self.action_size)
        self.buffer_size = 10
        self.batch_size = 4
        self.memory = ReplayBuffer(self.buffer_size)
        
        self.gamma = 0.99
        self.tau = 0.001
        self.last_state = None
        self.last_action = None
        
    def preprocess_state(self, state):
        """Reduce state vector to relevant dimensions."""
        return state[0:3]  # position only
      
    def postprocess_action(self, action):
        """Return complete action vector."""
        complete_action = np.zeros(self.task.action_space.shape)  # shape: (6,)
        complete_action[0:3] = action  # linear force only
        return complete_action

    def step(self, state, reward, done):
        # print ("STEPPING")
        # Choose an action
        state = self.preprocess_state(state)
        action = self.act(state)
        
        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.memory.add(self.last_state, self.last_action, reward, state, done)
            
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

        self.last_state = state
        self.last_action = action
        return self.postprocess_action(action)

    def act(self, states):
        print ("\nACTING")
        states = np.reshape(states, [-1, self.state_size])
        actions = self.actor_local.model.predict(states)
        print ("actions:")
        print (actions)
        return actions + self.noise.sample()

    def learn(self, experiences):
        # print ("LEARNING!!!!!!!")
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
        
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=None, theta=0.15, sigma=0.3):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state        

