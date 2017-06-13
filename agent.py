import numpy as np
import random
import tensorflow as tf

# Define all Global Variables
ACTOR_LR = 0.00001
CRITIC_LR = 0.001
ACTOR_TRAINING_EPOCH = 100
CRITIC_TRAINING_EPOCH = 100
SIGMA = 0.1

class Actor(object):

    def __init__(self, state_dim, action_dim, low, high):
        """
        Initiliasizes TF variables, loss function, advantage
        """
        tf.set_random_seed(1234)
        self.sess = tf.Session()
        self.state_dim = state_dim
        self. action_dim = action_dim

        self.state = tf.placeholder(tf.float32, [None, self.state_dim])
        with tf.variable_scope("Actor"):
            self.actor_output = self.build_network(self.state_dim, self.action_dim)
        
        #Placeholder for Actual actions taken
        #self.actions = tf.placeholder(tf.float32,[None, self.action_dim])

        self.normal_dist = tf.contrib.distributions.Normal(self.actor_output, SIGMA)

        self.action = self.normal_dist._sample_n(1)

        self.action = tf.clip_by_value(self.action,low, high)

        # Placeholder for advantages calculated from critic
        self.advantage = tf.placeholder(tf.float32,[None, 1])

        # Operation : Log of action taken
        #log_prob = tf.log(tf.clip_by_value(self.action, 1e-10, 1.0))

        # Operation : Log(a/s)*advantage
        #eligibility = log_prob*self.advantage

        # Defininikng Surrogate objective function
        #self.loss = -tf.reduce_sum(eligibility)

        self.loss = self.normal_dist.log_prob(self.action)*self.advantage

        self.loss = -tf.reduce_sum(self.loss)

        # Defining optimzier
        self.optimizer = tf.train.AdamOptimizer(ACTOR_LR).minimize(self.loss,
                                                                   global_step =
                                                                   tf.contrib.framework.get_global_step())

        #Initiliaze all variables
        self.sess.run(tf.global_variables_initializer())


    def build_network(self, state_dim, action_dim):
        """
        Creates Network
        """
        n_hidden_1 = 10
        n_hidden_2 = 10
        n_hidden_3 = 10

        weightsX={
            'h1':tf.get_variable('W1', shape = (self.state_dim, self.action_dim), initializer = tf.contrib.layers.xavier_initializer(), trainable = True),
                        
#            'h2':tf.get_variable('W2', shape = (n_hidden_1, n_hidden_2), initializer = tf.contrib.layers.xavier_initializer(), trainable = True),
#            
#            'h3':tf.get_variable('W3', shape = (n_hidden_2, n_hidden_3), initializer = tf.contrib.layers.xavier_initializer(), trainable = True),
#            
            'out':tf.get_variable('out', shape = (n_hidden_3, self.action_dim), initializer = tf.contrib.layers.xavier_initializer(), trainable = True)
        }

        biasesX={
            'b1':tf.get_variable('B1', shape = (self.action_dim), initializer = tf.contrib.layers.xavier_initializer(), trainable = True), 
#            'b2':tf.get_variable('B2', shape = (n_hidden_2), initializer = tf.contrib.layers.xavier_initializer(), trainable = True),
#            'b3':tf.get_variable('B3', shape = (n_hidden_3), initializer = tf.contrib.layers.xavier_initializer(), trainable = True),
#        
            'out':tf.get_variable('outB', shape = (self.action_dim), initializer = tf.contrib.layers.xavier_initializer(), trainable = True),
        }

        #hidden layer 1
        layer1=tf.add(tf.matmul(self.state,weightsX['h1']),biasesX['b1'])
        out_layer=(layer1)

        #hidden layer 2
        #layer2=tf.add(tf.matmul(layer1,weightsX['h2']),biasesX['b2'])
        #layer2=tf.nn.tanh(layer2)

        ##hidden layer 3
        #layer3=tf.add(tf.matmul(layer2,weightsX['h3']),biasesX['b3'])
        #layer3=tf.nn.tanh(layer3)

        #Output layer
        #out_layer = tf.matmul(layer1, weightsX['out']) + biasesX['out']
        #out_layer = tf.nn.tanh(out_layer)

        return out_layer

    def predict(self, state):
        """
        Forward pass pf Neural Network 
        predicts mean and standard deviation of continuous actions
        """
        #print "Predicting..."
        return self.sess.run(self.action, feed_dict = {
            self.state : state
        })


    def train(self, states, actions, advantages):
        """
        Trains the neural network with actions and advantages
        """
        for _ in range(ACTOR_TRAINING_EPOCH):
            _, c = self.sess.run([self.optimizer, self.loss], feed_dict = {
                self.state : states, 
                #self.action : actions,
                self.advantage : advantages
            })

class Critic(object):

      
    def __init__(self, state_dim, action_dim):
        """  
        Initiliasizes TF variables, loss function, advantage
        """

        tf.set_random_seed(1234)
        self.sess = tf.Session()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state = tf.placeholder(tf.float32, [None, self.state_dim])

        with tf.variable_scope("Critic"):
            # get outputs from critic network
            self.critic_output = self.build_network(self.state_dim,1)
        # Placeholder for target values
        self.target = tf.placeholder(tf.float32, [None, 1])

        # Calculate mean square error loss
        #self.loss = tf.reduce_mean(tf.squared_difference(self.critic_output,self.target))

        self.loss = tf.nn.l2_loss(self.critic_output - self.target)

        # Define Optimizer
        self.optimizer = \
        tf.train.AdamOptimizer(CRITIC_LR).minimize(self.loss,global_step = tf.contrib.framework.get_global_step())

        # Initilize all tf vars
        self.sess.run(tf.global_variables_initializer())

    def build_network(self, state_dim, action_dim):
        """
        Creates Network
        """

        n_hidden_1 = 10
        n_hidden_2 = 10
        n_hidden_3 = 10

        weightsX={
            'h1':tf.get_variable('W1', shape = (self.state_dim, n_hidden_1), initializer = tf.contrib.layers.xavier_initializer(), trainable = True),
                        
#            'h2':tf.get_variable('W2', shape = (n_hidden_1, n_hidden_2), initializer = tf.contrib.layers.xavier_initializer(), trainable = True),
#            
#            'h3':tf.get_variable('W3', shape = (n_hidden_2, n_hidden_3), initializer = tf.contrib.layers.xavier_initializer(), trainable = True),
#            
            'out':tf.get_variable('out', shape = (n_hidden_1, self.action_dim), initializer = tf.contrib.layers.xavier_initializer(), trainable = True)
        }

        biasesX={
            'b1':tf.get_variable('B1', shape = (n_hidden_1), initializer = tf.contrib.layers.xavier_initializer(), trainable = True), 
#            'b2':tf.get_variable('B2', shape = (n_hidden_2), initializer = tf.contrib.layers.xavier_initializer(), trainable = True),
#            'b3':tf.get_variable('B3', shape = (n_hidden_3), initializer = tf.contrib.layers.xavier_initializer(), trainable = True),
#        
            'out':tf.get_variable('outB', shape = (self.action_dim), initializer = tf.contrib.layers.xavier_initializer(), trainable = True),
        }

        #hidden layer 1
        layer1=tf.add(tf.matmul(self.state,weightsX['h1']),biasesX['b1'])
        layer1=tf.nn.relu(layer1)

        #hidden layer 2
#        layer2=tf.add(tf.matmul(layer1,weightsX['h2']),biasesX['b2'])
#        layer2=tf.nn.relu(layer2)
#
#        #hidden layer 3
#        layer3=tf.add(tf.matmul(layer2,weightsX['h3']),biasesX['b3'])
#        layer3=tf.nn.relu(layer3)

        #Output layer
        out_layer = tf.matmul(layer1, weightsX['out']) + biasesX['out']

        return out_layer

    def getValue(self, state):
        """
        input : State
        output : Estimated value of the state
        """

        return self.sess.run(self.critic_output, feed_dict = {
            self.state : state
        })


    def train(self, states, targets):
        """
        Trains for TD Loss in Value function... baseline idea
        """
        self.sess.run([self.optimizer, self.loss], feed_dict = {
            self.state : states,
            self.target : targets
        })

class PGAgent(object):

    def __init__(self, state_dim, action_dim, gamma, low, high ):
        """
        Initializes all physical dimensions
        """
        # Get Dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Instantiate actor and critic
        self.actor = Actor(self.state_dim, self.action_dim, low, high)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.gamma = gamma
        self._steps = 0


    def select_action(self, state):
        """
        Selects actions with given mean and sigma
        action variable = [2x1] vector with mean and standard deviation
        select continous action with gaussian with mean and std
        """
        action = self.actor.predict(state)

        return action

    def finish_episode(self, transitions):
        """
        Collects all transitons, finds advantages, finds discounted sum of rewards 
        Calculates Policy gradients, Calculates Targets


        input : transitions -> List of List :[state, action, reward, next_state, terminal]
        """
        print "Training ################___________________"

        # Get Cumulative reward
        for i in range(len(transitions)):
            if transitions[i][4] == 0:
                #print "State : ", transitions[i][0]
                #print "Next State: ", transitions[i][3]
                #print "Value : ", self.critic.getValue(transitions[i][3])
                transitions[i].append(transitions[i][2] + self.gamma*self.critic.getValue(transitions[i][3]))
            else:
                transitions[i].append(transitions[i][2])

        # transitions is [s, a, r, s1, terminal, R]

        # Get Baseline
        for i in range(len(transitions)):
            transitions[i].append(self.critic.getValue(transitions[i][0]))

        # transitions is [s, a, r, s1, terminal, R, b]

        # Get Advantages
        for i in range(len(transitions)):
            transitions[i].append(transitions[i][5] - transitions[i][6])
        
        # transitions is [s, a, r, s1, terminal, R, b, adv]

        # Training critic
        # Gather states
        b_states = np.zeros((len(transitions), self.state_dim))
        b_targets = np.zeros((len(transitions), 1))
        b_actions = np.zeros((len(transitions), 1))
        b_advantages =  np.zeros((len(transitions), 1))

        for i in range(len(transitions)):
            b_states[i] = transitions[i][0]
            b_targets[i] = transitions[i][5]
            b_actions[i] = transitions[i][1]
            b_advantages[i] = transitions[i][7]

        #print "Advantages ",b_advantages.shape

        self.critic.train(b_states, b_targets)

        self.actor.train(b_states, b_actions, b_advantages)

        self._steps = 0

