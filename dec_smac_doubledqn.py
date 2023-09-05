import numpy as np
import tensorflow as tf
from tensorflow import keras
import sonnet as snt
from smac_utilities import Agent

		
class Cohort(Agent):

	def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
			input_dims, epsilon_dec=1e-4, epsilon_end=0.01,
			mem_size=1e6, fname='doubledqn_model.h5'):
		super().__init__(lr, gamma, n_actions, epsilon, batch_size,
			input_dims, epsilon_dec, epsilon_end,
			mem_size, fname)
		
	def choose_action(self, observation, avail_acts_oh):
		avail_acts = np.nonzero(avail_acts_oh)[0]
		if np.random.random() < self.epsilon:
			action = np.random.choice(avail_acts)
		else:
			# https://stackoverflow.com/questions/64520917/valueerror-error-when-checking-input-expected-dense-1-input-to-have-shape-8
			state = np.reshape(np.array(observation), (1,len(observation)))
			actions = self.q_online.__call__(state).numpy()[0]
			
			subset_idx = np.argmax(actions[avail_acts])
			action = avail_acts[subset_idx]
			
		return action