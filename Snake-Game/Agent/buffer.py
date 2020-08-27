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

		if done:
			self.discount_rewards()

	def store_policies(self, pi, pi_k):
		self.policies.append(pi)
		self.prev_policies.append(pi_k)

	def get_trajectories(self):

		s = torch.Tensor(self.states).float()
		s_p = torch.Tensor(self.prev_states).float()
		a = torch.Tensor(self.actions).float()
		r = torch.Tensor(self.rewards).float()
		d = torch.Tensor(self.terminals).float()

		return s, s_p, a, r, d

	def get_policies(self):
		return None

	def discount_rewards(self):

		# get interval between terminal and last terminal
		
		pass


