import torch
from tqdm import tqdm
import numpy as np

class Network(torch.nn.Module):

    def __init__(self, in_dim, out_dim, hid_dim, num_lay=1, lr=10e-1):

        super(Network, self).__init__()

        self.define_network(in_dim, out_dim, hid_dim, num_lay)

        self.loss = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if torch.cuda.is_available():
             self.device = torch.device("cuda:0")
        else: self.device = torch.device("cpu:0")
        self.to(device=self.device)

        self.historical_loss = []
        self.historical_accuracy = []

    def define_network(self, in_dim, out_dim, hid_dim, num_lay):

        self.lstm = torch.nn.LSTM(
            input_size=in_dim,
            hidden_size=hid_dim,
            num_layers=num_lay)

        self.fc = torch.nn.Linear(1, 1)

        self.sigmoid = torch.nn.Sigmoid()
        self.leaky_relu = torch.nn.LeakyReLU()


    def forward(self, x):

        out = torch.Tensor(x).to(self.device)

        #print(out.shape)

        out = self.lstm(out)

        #print(out[0][-1].shape)

        out = self.fc(out[0][-1].reshape(-1, 1))
        out = self.sigmoid(out)

        #print(out.shape)

        return out.to(torch.device("cpu:0"))

    def optimize(self, x, y, batch_sz, iters):
        
        x = torch.Tensor(x.reshape([x.shape[1], x.shape[0], 1]))
        y = torch.Tensor(y)

        print(x.shape)
        print(y.shape)

        num_batches = y.shape[0]//batch_sz

        for i in tqdm(range(iters), "Training Network"):
        
            for b in range(num_batches):
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                p = self.forward(x[:,b*batch_sz:(b+1)*batch_sz,:])
                self.accuracy(p, y[b*batch_sz:(b+1)*batch_sz, :])
                loss = self.loss(p, y[b*batch_sz:(b+1)*batch_sz, :])
                self.historical_loss.append(loss.detach())
                loss.backward()
                self.optimizer.step()
    

    def accuracy(self, p, y):

        c = (p.clone().detach()>0.5).float()

        a = torch.abs(c-y.clone().detach())

        self.historical_accuracy.append(1-torch.sum(a).item()/y.shape[0])


def main():

    x = np.random.randint(0, high=2, size=[1000, 50], dtype=int)
    y = np.random.randint(0, high=2, size=[1000, 1], dtype=int)
    net = torch.nn.LSTM(1, 10, 1)
    p = net.forward(x.reshape([50, 1000, 1]))

    print(len(p))
    print(p[0].shape)
    print(p[0][0].shape)

if __name__ == "__main__":

    main()