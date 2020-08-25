import torch
import numpy as np 

from .policy import Policy
from .value import Value
from .buffer import Buffer


class PPO(object):

	def __init__(self):
		
		self.policy = Policy()
		self.value = Value()
		self.buffer = Buffer()

	def calc_advantage(self):
		
		v1 = self.value()
		v2 = self.value()

		return v2 - v1


	def act(self, state):
		

		p = self.policy(state)
		old_p = self.old_policy(state)


		return action


	def update(self):


		self.value.optimize(x, y)
		self.policy.optimize(x, y)

def main():
	pass

if __name__ == "__main__":
	main()