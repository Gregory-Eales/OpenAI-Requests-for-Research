import torch
import numpy as np 

from .policy import Policy
from .value import Value
from .buffer import Buffer


class PPO(object):

	def __init__(self):
		
		self.policy = Policy(in_channel=1, hidd_channel=16, kernal_sz=2, lin_dim=5, alpha=1e-3)
		self.value = Value(in_channel=1, hidd_channel=16, kernal_sz=2, lin_dim=5, alpha=1e-3)
		self.buffer = Buffer()

	def calc_advantage(self):
		
		v1 = self.value()
		v2 = self.value()

		return v2 - v1


	def act(self, state):
	
		p = self.policy(state)
		old_p = self.old_policy(state)

		return action

	def store(self, state, prev_state, action, reward, done):
		self.buffer.store(state, prev_state, action, reward, done)

	def update(self):

		s, s_p, a, r, d = self.buffer.get_trajectories()
		pi, pi_k = self.buffer.get_policies()

		self.value.optimize(x, y)
		self.policy.optimize(x, y)


def main():
	pass

if __name__ == "__main__":
	main()