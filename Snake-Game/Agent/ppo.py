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
		pass


	def get