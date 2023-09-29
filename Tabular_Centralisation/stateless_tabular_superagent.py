import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import copy

reward = np.array([11,-30,0,-30,7,6,0,0,5])

# parameters
alpha = 0.2
eps = 0.1
n_epi = 1000
n_agents = 2
n_actions = 3**n_agents
actions = np.array(range(n_actions))

#input to be evaluated
Q = np.random.rand(n_actions)

#store Qs
Q_storage = np.zeros((n_epi,n_actions))

#Q-learning algorithm
for i in range(n_epi):
    #each agent chooses an action
    A = epsilon_greedy(eps,Q,actions)
        
    #observe R
    r = reward[A]
    Q[A] = Q[A] + alpha*(r-Q[A])

    Q_storage[i,:] = Q

print(Q)
for a in range(n_actions):
    plt.plot(Q_storage[:,a])
plt.show()