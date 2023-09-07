import numpy as np
import tensorflow as tf
from tensorflow import keras
import sonnet as snt
from replaybuffers import *

class GradientFarm():
	def __init__(self,slen,s_examples):
		#get tuples, store list (per action gradient thingy)
		self.memory = {}
		self.n_sufficient_states = 0
		self.sufficientlen = slen
		self.sufficient_examples = s_examples
		self.ready_to_process = False

	def harvest(self,state_unmutable,action_grads):
		try: 
			self.memory[state_unmutable].append(action_grads)

			#track the full slots
			if len(self.memory[state_unmutable])>self.sufficientlen: 
				self.sufficients+=1
				if self.sufficients>self.sufficient_examples:
					self.ready_to_process = True

		except:
			self.memory[state_unmutable] = [action_grads]
			
		
	def boil_down(self):
		


		return