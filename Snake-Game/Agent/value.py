import torch

class Value(torch.nn.Module):

	def __init__(self, in_channel, hidd_channel, kernal_sz, lin_dim, alpha=1e-3):

		super(Value, self).__init__()

		self.in_channel = in_channel
		self.hidd_channel = hidd_channel
		self.kernal_sz = kernal_sz
		self.lin_dim = lin_dim

		self.conv1 = torch.nn.Conv2d(
			in_channels=in_channel,
			out_channels=hidd_channel,
			kernel_size=kernal_sz
			)

		self.conv2 = torch.nn.Conv2d(
			in_channels=hidd_channel,
			out_channels=hidd_channel,
			kernel_size=kernal_sz
			)

		self.conv3 = torch.nn.Conv2d(
			in_channels=hidd_channel,
			out_channels=hidd_channel,
			kernel_size=kernal_sz
			)

		self.l1 = torch.nn.Linear(lin_dim, 4)

		self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)

		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
		self.to(self.device)


	def loss(self, pi, pi_k, epsilon, advantage):

		term1 = pi/pi_k * advantage
		term2 = torch.clamp(advantage, min=1-epsilon, max=1+epsilon)

		return torch.sum(torch.min(term1, term2))


	def forward(self, x):

		out = torch.Tensor(x).to(self.device)
		
		out = self.conv1(out)
		out = self.conv2(out)
		out = self.conv2(out)

		return out.to(torch.device('cpu:0'))


	def optimize(self, x, y, epochs=10, batch_sz=16):


		num_batch = x.shape[0]//batch_sz

		for e in tqdm(range(epochs)):

			for b in range(num_batch):
				
				p = self.forward(x[b*batch_sz:(b+1)*(batch_sz)])