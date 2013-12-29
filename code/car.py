from pylab import *
import numpy as np

class car:
    
    def __init__(self):

        # setup your parameters here.

        # sigma place neurons
        self.c_s_sigma_p = float(1) / 30

        # sigma velocity neurons
        self.c_s_sigma_v = 0.2

        # eligibility trace decay parameter
        self.c_s_lambda = 0.95

        # greedy probability
        self.c_s_greedy = 0.1

        # learning rate
        self.c_s_eta = 0.005

        # reward discount
        self.c_s_gamma = 0.95

        # current Q value
        self.c_s_current_q = 0

        # last Q value
        self.c_s_next_q = 0
        
        # Place cells: 31 x 31
        # Velocity cells: 11 x 11
        # output cells: 9
        
        # There are 9 different possible actions
        self.c_v_actions = np.arange(0,9)

        # There have 31 x 31 place neurons
        self.c_v_place = np.linspace(0,1,31)

        # There are 11 x 11 velocity neurons
        self.c_v_velocity = np.linspace(-1,1,11)

        # The weight vectors for the 9 different output neurons
        self.c_m_weights = np.zeros([9,31**2 + 11**2])

        # eligibility trace vector for the 9 different output neurons
        self.c_m_eligibility = np.zeros([9,31**2 + 11**2])

    def reset(self):
    
        # reset values before each trial.
        self.time = 0

        self.c_m_eligibility = np.zeros([9,31**2 + 11**2])

        self.c_s_next_q = 0

        self.c_s_current_q = 0

    # computes the activations at place neurons
    def getPlaceActivations(self, s_x, s_y):
        #print("= P =")
        v_x = np.array([self.c_v_place, ]*31).transpose().flatten()
        v_y = np.array([self.c_v_place,]*31).flatten()

        return self.getActivations(s_x, s_y, self.c_s_sigma_p, v_x, v_y)

    # Computes the activations at velocity neurons
    def getVelocityActivations(self, s_x, s_y):
        v_x = np.array([self.c_v_velocity, ]*11).transpose().flatten()
        v_y = np.array([self.c_v_velocity,]*11).flatten()

        return self.getActivations(s_x, s_y, self.c_s_sigma_v, v_x, v_y)

    # Computes the activations of all neurons (place or velocity neurons)
    def getActivations(self, s_x, s_y, s_sigma, v_neuron_x, v_neuron_y):

        # gaussian activation function
        f_activation = lambda x,y: np.exp(-((s_x - x)**2 + (s_y - y)**2)/(2*(s_sigma**2)))

        activation = f_activation(v_neuron_x, v_neuron_y)

        return activation

    # compute the eligibility trace for the current step
    def updateEligibilityTrace(self, s_action, v_place_activation, v_velocity_activation):
        
        # add activations to active action
        self.c_m_eligibility[s_action] += np.concatenate((v_place_activation, v_velocity_activation))

        # decay eligibility
        self.c_m_eligibility *= self.c_s_lambda * self.c_s_gamma

        #print self.c_m_eligibility

    # update the weights using delta, the eligibility trace and the learning rate
    def updateWeights(self, s_reward):

        s_delta = s_reward - self.c_s_current_q + self.c_s_gamma * self.c_s_next_q

        self.c_m_weights = self.c_m_weights + self.c_s_eta * s_delta * self.c_m_eligibility

        #print self.c_m_weights

    def directionPlot(self):
        f_x = lambda a: np.cos(-2*np.pi*a / 8 + np.pi / 2)
        f_y = lambda a: np.sin(-2*np.pi*a / 8 + np.pi / 2)

        max_actions = np.argmax(self.c_m_weights[0:9, 0:31*31], axis=0)
        u = f_x(max_actions).reshape([31,31])
        v = f_y(max_actions).reshape([31,31])

        figure(300)
        quiver(u,v)

    def printWeights(self):
        figure()
        imshow(self.c_m_weights)

    # Computes all Q(s,a) values for all actions at state s. Returns best Q(s,a)
    # and its corresponding action a.
    def getBestQValues(self, v_place_activations, v_velocity_activations):

        v_q = np.zeros(9)

        for a in self.c_v_actions:
            v_q[a] = np.inner(np.concatenate((v_place_activations, v_velocity_activations)), self.c_m_weights[a])

        # choose action with 1-e greedy policy
        greedy = np.random.choice([True, False], p=[1-self.c_s_greedy, self.c_s_greedy])

        # find maximum
        s_best_q = v_q.max()
        s_best_action = np.random.choice(np.nonzero(v_q == s_best_q)[0])

        if greedy:
            s_q = s_best_q
            s_action = s_best_action
        else:
            # select randomly (with uniform distribution) one action
            s_action = np.random.choice(np.delete(self.c_v_actions, s_best_action))
            s_q = v_q[s_action]

        return (s_q, s_action)


    def choose_action(self, position, velocity, R, learn = True):
        # This method must:
        # -choose your action based on the current position and velocity.
        # -update your parameters according to SARSA. This step can be turned off by the parameter 'learn=False'.
        #
        # The [x,y] values of the position are always between 0 and 1.
        # The [vx,vy] values of the velocity are always between -1 and 1.
        # The reward from the last action R is a real number
    	
    	# add your action choice algorithm here

        v_place_activations = self.getPlaceActivations(position[0], position[1])
        v_velocity_activations = self.getVelocityActivations(velocity[0], velocity[1])

        (self.c_s_next_q, action) = self.getBestQValues(v_place_activations, v_velocity_activations)

        # print("Action: {0}".format(action))

        if learn:    

            # update weights
            self.updateWeights(R)

            # update eligibility trace
            self.updateEligibilityTrace(action, v_place_activations, v_velocity_activations)

            # next q is current in next iteration
            self.c_s_current_q = self.c_s_next_q
            
    	self.time += 1

    	return action