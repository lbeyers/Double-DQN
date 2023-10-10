import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import copy
from exploration_strats import *

reward = np.array([[11,-30,0],[-30,7,6],[0,0,5]])

# parameters
alpha = 0.2
eps = 0.1
n_epi = 100000
n_actions = 3
n_agents = 2
actions = np.array(range(n_actions))

#input to be evaluated
Q = np.random.rand(n_agents,n_actions)

#Q-learning algorithm
for i in range(n_epi):
    #each agent chooses an action
    actions_taken = np.zeros(n_agents).astype(int)
    for a_id in range(n_agents):
        #choose A
        actions_taken[a_id] = epsilon_greedy(eps,Q,actions,a_id)
        #actions_taken[a_id] = boltzmann(i,Q,actions,a_id)
        #actions_taken[a_id] = contrived(i,Q,actions,a_id)
        
    #observe R
    r = reward[tuple(actions_taken)]

    for a_id in range(n_agents):
        #update Q
        Q[a_id,actions_taken[a_id]] = Q[a_id,actions_taken[a_id]] + alpha*(r-Q[a_id,actions_taken[a_id]])

print(Q)