import numpy as np
import os
import pandas as pd
from quad_controller_rl import util
from quad_controller_rl.agents.base_agent import BaseAgent
import keras
from keras import layers, models, optimizers
from keras import backend as K
import random
from collections import namedtuple
import heapq

Experience_Replay = namedtuple("Experience_replay",
                               field_names=['state', 'action', 'reward', 'next_state', 'done'])


#############
#   ACTOR   #
#############

class Actor:
    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size = state_size                                # SET CORRESPONDING VARIABLES
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low      # SET THE RANGE
        self.build_NN()                                             # BUILD THE MODEL

    def build_NN(self):
        states = layers.Input(shape=(self.state_size,), name='states')    # INPUT
        net = layers.Dense(units=32, activation=None)(states)             # FIRST LAYER
        # net = layers.BatchNormalization()(net)                            # NORMALIZE
        net = layers.Activation('relu')(net)                              # ACTIVATION FUNCTION
        net = layers.Dense(units=64, activation=None)(net)                # SECOND LAYER
        # net = layers.BatchNormalization()(net)                            # NORMALIZE
        net = layers.Activation('relu')(net)                              # ACTIVATION
        net = layers.Dense(units=32, activation='relu')(net)              # THIRD LAYER, NO NORMALIZE 
        
        raw_actions = layers.Dense(units=self.action_size,                # ACTION SIZE IS OUTPUT
                                   activation='sigmoid',                  # SIGMOID TO SQUISH
                                   name='raw_actions')(net)               # BETWEEN [0.0, 1.0]
        
        # re scale the values to appropriate range
        actions = layers.Lambda(lambda x: (x * self.action_range) +self.action_low,
                                name='actions')(raw_actions)
        
        self.model = models.Model(inputs=states, outputs=actions)         # create the model
        
        action_gradients = layers.Input(shape=(self.action_size,))        # define input for action gradients
        loss = K.mean(-action_gradients*actions)                          # define loss function
        
        optimizer = optimizers.Adam(lr=0.0005)                            # define optimizer (lower learning rate??)
        updates_op = optimizer.get_updates(                               # define training function
            params=self.model.trainable_weights,
            loss=loss)
        
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)

#############
#   CRITIC  #
#############

class Critic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size                # define state size (should be 2)
        self.action_size = action_size              # define action size (should be 1)
        self.build_NN()                             # build the NN model
        
    def build_NN(self):                                                      
        # Try to mimic Actor ??
        states = layers.Input(shape=(self.state_size,), name='states')               # define states input
        net_states = layers.Dense(units=32, activation=None)(states)               # first layer 
        # net_states = layers.BatchNormalization()(net_states)                         # first normalize
        net_states = layers.Activation('relu')(net_states)                           # first activation function
        net_states = layers.Dense(units=64, activation=None)(net_states)           # second layer 
        # net_states = layers.BatchNormalization()(net_states)                         # second normalize
        net_states = layers.Activation('relu')(net_states)                           # second activation function
        # net_states = layers.Dense(units=16, activation='relu')(net_states)           # third layer, no activation

        # try to mimic Actor again ??
        actions = layers.Input(shape=(self.action_size,), name='actions')            # define inputs
        net_actions = layers.Dense(units=32, activation=None)(actions)             # first layer 
        # net_actions = layers.BatchNormalization()(net_actions)                       # first normalize
        net_actions = layers.Activation('relu')(net_actions)                         # first activation function
        net_actions = layers.Dense(units=64, activation=None)(net_actions)         # second layer 
        # net_actions = layers.BatchNormalization()(net_actions)                       # second normalize
        net_actions = layers.Activation('relu')(net_actions)                         # second activation function
        # net_actions = layers.Dense(units=16, activation='relu')(net_actions)         # third layer, no activation
        
        # combination
        net = layers.Add()([net_states, net_actions])                                # combine state and actions
        net = layers.Activation('relu')(net)                                         # fourth activation
        # net = layers.Dense(units=16, activation='relu')(actions)                     # fifth activation

        Q_values = layers.Dense(units=1, name='q_values')(net)                       # generate a Q_value

        self.model = models.Model(inputs=[states, actions], outputs=Q_values)        # create the model

        optimizer = optimizers.Adam(lr=0.0005)                              # set the optimizer
        self.model.compile(optimizer=optimizer, loss='mse')                 # compile the model

        action_gradients = K.gradients(Q_values, actions)                   # calculate the gradients with respect to actions

        # function to get the action gradients
        # to be used by Actor
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)


###################
# PRIORITY BUFFER #
###################

'''
class Pair(object):
    def __init__(self, reward, state, action, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
    def __lt__(self, other):
        return self.reward < other.reward
'''
        
###################
#  REPLAY BUFFER  #
###################

class ReplayBuffer:
    def __init__(self, size=1000):
        self.size = size
        self.memory = []
        self.idx = 0
    def add(self, state, action, reward, next_state, done):
        # set = Pair(reward, state, action, next_state, done)
        set = Experience_Replay(state, action, reward, next_state, done)
        if len(self.memory) < self.size:
            # heapq.heappush(self.memory, set)
            self.memory.append(set)
        else:
            # trash = heapq.heappushpop(self.memory, set)
            self.memory[self.idx] = set
            self.idx = (self.idx + 1) % self.size
    def sample(self, batch_size=64):
        return random.sample(self.memory, batch_size)
    def reset(self):
        self.memory[:] = []
    def __len__(self):
        return len(self.memory)

      
class Combined_DDPG_Policy(BaseAgent):
    def __init__(self, task):
        self.task = task                            # task passed to agent
        self.state_size = 3                         # now includes which state (takeoff=0 or landing=1)

        self.state_low = self.task.observation_space.low[0:3]      # min position x,y,z
        self.state_high = self.task.observation_space.high[0:3]    # max position x,y,z

        self.state_range = self.state_high - self.state_low                      # position ranges
        self.action_size = 1                                                     # constrained only to z direction
        self.action_low = task.action_space.low[2]                               # lowest z-force (-25.0)
        self.action_high = task.action_space.high[2]                             # highest z-force (25.0)
        self.action_range = self.action_high - self.action_low                   # range (50.0)
        print("Original spaces: {}, {}\nConstrained spaces: {}, {}".format(
            self.task.observation_space.shape, self.task.action_space.shape,
            self.state_size, self.action_size))                                  # Debug
        
        ################
        # SAVE WEIGHTS #
        ################
        self.load_weights = True
        self.save_weights_every = 4
        self.model_dir = util.get_param('out')
        self.model_name = "MODEL_WEIGHTS"
        self.model_ext = ".h5"
        if self.load_weights or self.save_weights_every:
            # Define Actor weights h5 file
            self.actor_filename = os.path.join(self.model_dir, util.get_param('task'),
                "{}_actor{}".format(self.model_name, self.model_ext))
            # Define Critic weights h5 file
            self.critic_filename = os.path.join(self.model_dir, util.get_param('task'),
                "{}_critic{}".format(self.model_name, self.model_ext))
            # Debug print statements
            print("Actor filename :", self.actor_filename)  # [debug]
            print("Critic filename:", self.critic_filename)  # [debug]
        
        ###################################
        # CREATE ACTOR AND CRITIC OBJECTS #
        ###################################
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        
        ################################################
        # Load pre-trained model weights, if available #
        ################################################
        if self.load_weights and os.path.isfile(self.actor_filename):
            try:
                self.actor_local.model.load_weights(self.actor_filename)
                self.critic_local.model.load_weights(self.critic_filename)
                print("Model weights loaded from file!")  # [debug]
            except Exception as e:
                print("Unable to load model weights from file!")
                print("{}: {}".format(e.__class__.__name__, str(e)))
        if self.save_weights_every:
            print("Saving model weights", "every {} episodes".format(
                self.save_weights_every) if self.save_weights_every else "disabled")  # [debug]
        
        ###############
        # SET WEIGHTS #
        ###############
        # target model gets weights from local model
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())

        ####################
        # BUFFER VARIABLES #
        ####################
        self.buffer_size = 50000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size)
        
        self.noise = OUNoise(self.action_size)
        self.gamma = 0.99                       # Gamma stays at 0.99
        self.tau = 0.01                         # changed TAU to 0.01
        
        self.last_state = None
        self.last_action = None
        self.reward_vector = []
        self.episode_count = 0
        self.step_count = 0


        self.episode = 0                        # Episode number for given state
        self.episode_total = 0                  # Total number of episodes

        ###############################
        # SAVE EP REWARDS TO CSV FILE #         # **WARNING!! STATS_COLUMNS HAS CHANGED**
        ###############################
        self.total_reward = 0
        self.stats_filename = os.path.join(
            util.get_param('out'), util.get_param('task'), "stats_{}.csv".format(util.get_timestamp()))
        self.stats_columns = ['task', 'episode', 'total_reward']
        self.episode_num = 1
        print("Saving stats {} to {}".format(self.stats_columns, self.stats_filename))
        
    ###########################
    #   RESET STEP VARIABLES  #
    ###########################
        
    def reset(self):
        self.last_action = None
        self.last_state = None
        self.total_reward = 0
        self.step_count = 0
        
    ####################################################
    #   METHODS USED BY TASK TO CHANGE BETWEEN TASKS   #
    ####################################################
    
    def get_epCount(self):
        return self.episode
      
    def reset_epCount(self):
        print ("Episodes have RESET")
        self.episode = 0
        
    ###########################
    #   WRITE STATS TO CSV    #
    ###########################
        
    def write_stats(self, stats):
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)
        df_stats.to_csv(self.stats_filename,
                        mode='a',
                        index=False,
                        header=not os.path.isfile(self.stats_filename))
        
    ###########################
    #   PREPROCESS STATE      #
    ###########################
        
    def preprocess_state(self, state, task):
        return np.array([state[2], state[9], task])  # position, velocity, and which_state
      
    ###########################
    #   POSTPROCESS STATE     #
    ###########################
      
    def postprocess_action(self, action):
        complete_action = np.zeros(self.task.action_space.shape)
        complete_action[2] = action
        '''
        if self.step_count % 100 == 0:
            print ("Action to be processed:\n{0}".format(action))
            print ("Processed action:\n{0}".format(complete_action))
        '''
        return complete_action

    ###########################
    #   STEP                  #
    ###########################
    
    def step(self, state, reward, done):
        which_task = self.task.get_which_task()            # Determine if taking off or landing
        self.total_reward += reward                        # add reward to total
        self.step_count += 1                               # increase step count
        state = self.preprocess_state(state, which_task)   # convert state to only necessary components

        # normalize position between [0, 1]
        state[0] = (state[0]-self.state_low[0])/(self.state_high[0]-self.state_low[0])
        
        state = state.reshape(1, -1)                  # convert to row vector
        action = self.act(state)                      # call act method with current state

        # Add <LS, LA, R, S, D> to replay buffer
        if self.last_state is not None and self.last_action is not None:
            self.memory.add(state=self.last_state, action=self.last_action, reward=reward, next_state=state, done=done)
        
        # Start Batch learning when possible    
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
            
        
        self.last_state = state           # Set last state before reset
        self.last_action = action         # Set last action before reset
        
        ##############
        #   IF DONE  #
        ##############
        
        if done:
            if self.save_weights_every and self.episode % self.save_weights_every == 0:
                self.actor_local.model.save_weights(self.actor_filename)
                self.critic_local.model.save_weights(self.critic_filename)
                print("Model weights saved at episode", self.episode_num)  # [debug]
            # calc average reward per action for given episode
            avg_reward = self.total_reward / self.step_count
            # WRITE STATS TO FILE
            self.write_stats([which_task, self.episode_total, self.total_reward])
            # print statements for debugging
            print ("which_task?   :\t{0}".format(which_task))
            print ("self.episode  :\t{0}".format(self.episode))
            print ("Total Reward  :\t{0}".format(self.total_reward))
            print ("Average Reward:\t{0}".format(self.total_reward/self.step_count))
            self.episode_total += 1
            self.episode += 1
            self.reset()

        return self.postprocess_action(action)    # return the action

    ######################
    #   ACT METHOD       #
    ######################
    
    def act(self, states):
        states = np.reshape(states, [-1, self.state_size])
        actions = self.actor_local.model.predict(states)
        noise = self.noise.sample()
        return actions + noise
      
    ########################
    #   LEARN METHOD       #
    ########################

    def learn(self, experiences):
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
        
        
    ######################
    #   SOFT UPDATE      #
    ######################

    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
        
######################
#   NOISE MODEL      #
######################
        
class OUNoise:
    def __init__(self, size, mu=None, theta=0.15, sigma=0.75):
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        self.state = self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
