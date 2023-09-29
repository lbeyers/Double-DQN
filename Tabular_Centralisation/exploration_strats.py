import numpy as np

def epsilon_greedy(eps,Q,actions,agent_id):
    # decide how to act
    if np.random.rand()<eps:
        # randomly choose action
        A = np.random.choice(actions)
    else:
        # use Q
        A = np.argmax(Q[agent_id,:])
    return A

def epsilon_greedy(eps,Q,actions):
    # decide how to act
    if np.random.rand()<eps:
        # randomly choose action
        A = np.random.choice(actions)
    else:
        # use Q
        A = np.argmax(Q)
    return A

def boltzmann(eps,Q,actions):
    T = 16
    probs = np.zeros(len(actions))
    for action_id in len(actions):
        probs[action_id] = np.e**Q[action_id]/T

    probs = probs/np.sum(probs)

    A = np.random.choice(actions,p=probs)
    return A