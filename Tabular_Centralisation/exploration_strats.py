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

# def epsilon_greedy(eps,Q,actions):
#     # decide how to act
#     if np.random.rand()<eps:
#         # randomly choose action
#         A = np.random.choice(actions)
#     else:
#         # use Q
#         A = np.argmax(Q)
#     return A

def boltzmann(i,Q,actions):
    T = 16*0.99**i
    probs = np.zeros(len(actions))
    for action_id in range(len(actions)):
        probs[action_id] = np.e**Q[action_id]/T

    probs = probs/np.sum(probs)

    A = np.random.choice(actions,p=probs)
    return A

def boltzmann(i,Q,actions,agent_id):
    T = 16*0.99**i
    probs = np.zeros(len(actions))
    for action_id in range(len(actions)):
        probs[action_id] = np.e**Q[agent_id,action_id]/T

    probs = probs/np.sum(probs)

    A = np.random.choice(actions,p=probs)
    return A

def contrived(i,Q,actions):
    reward = np.array([11,-30,0,-30,7,6,0,0,5])-12
    probs = reward/np.sum(reward)
    #probs = np.ones(9)/9

    A = np.random.choice(actions,p=probs)
    return A

# def contrived(i,Q,actions,agent_id):
#     #reward = np.array([11,-30,0,-30,7,6,0,0,5])-12
#     #probs = reward/np.sum(reward)
#     probs = np.ones(3)/3

#     A = np.random.choice(actions,p=probs)
#     return A