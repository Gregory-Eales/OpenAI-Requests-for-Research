import torch

class Network(torch.nn.Module):

    def __init__(self, in_size, hidden_size, out_size):

        super(Network, self).__init__()
        
        self.loss = torch.nn.MSELoss()


    def define_network(self):

        self.lstm = torch.nn.LSTM()
        self.leaky_relu = torch.nn.LeakyReLU()


    def forward(self, x):pass

    def optimize(self, x, y, batch_sz, iter):pass
