import torch
import numpy as np
from tqdm import tqdm



class Policy(torch.nn.Module):

	def __init__(self, in_dim, hidd_dim, out_dim, alpha=1e-3):

		
		self.in_dim = in_dim
		self.hidd_dim = hidd_dim
		self.out_dim = out_dim

		self.optimizer = torch.optim.Adam(self.parameters(), alpha=alpha)


	def forward(self):

		pass


	def optimize(self, x, y, epochs=10, batch_sz=16):


		num_batch = x.shape[0]//batch_sz

		for e in tqdm(range(epochs)):

			for b in range(num_batch):
				
				p = self.forward(x[b*batch_sz:(b+1)*(batch_sz)])


