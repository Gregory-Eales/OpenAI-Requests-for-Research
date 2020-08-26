import numpy as np
import torch

class Buffer(object):

	def __init__(self):

		self.reset()

	def reset(self):

		self.states = []
		self.prev_states = []
		self.rewards = []
		self.terminals = []
		self.actions = []

		self.policies = []
		self.prev_policies = []

	def store(self, state, prev_state, action, reward, done):

		self.states.append(state)
		self.prev_states.append(prev_state)
		self.actions.append(action)
		self.rewards.append(reward)
		self.terminals.append(done)

	def store_policies(self, pi, pi_k):
		self.policies.append(pi)
		self.prev_policies.append(pi_k)