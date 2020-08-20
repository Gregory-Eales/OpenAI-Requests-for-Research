import torch
from tqdm import tqdm
import numpy as np
from copy import copy

class Network(torch.nn.Module):

    def __init__(self, in_dim, out_dim, hid_dim, num_lay=1, lr=10e-1):

        super(Network, self).__init__()

        self.define_network(in_dim, out_dim, hid_dim, num_lay)

        self.loss = torch.nn.BCEWithLogitsLoss()
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

        self.fc = torch.nn.Linear(hid_dim, 1)

        self.sigmoid = torch.nn.Sigmoid()
        self.leaky_relu = torch.nn.LeakyReLU()


    def forward(self, x):

        out = torch.Tensor(x).to(self.device)

        #print(out.shape)

        out = self.lstm(out)

        #print(out[0][-1].shape)

        #print(out[0].shape)


        out = self.fc(out[0][0].reshape(x.shape[1], -1))
        out = self.sigmoid(out)

        #out = self.sigmoid(out[0][-1].reshape(-1, 1))

        #print(out.shape)

        return out.to(torch.device("cpu:0"))

    def optimize(self, x, y, batch_sz, iters):
        
        a = torch.Tensor(x)
        a = 1-torch.sum(a, dim=1, keepdim=True)%2
        print(a.shape)
        print()

        #print(a[a%2 == 0])
        print(a)
        print("ACC: ", self.accuracy(a, torch.Tensor(y)))

        x = torch.Tensor(x.reshape([x.shape[1], x.shape[0], 1]))
        y = torch.Tensor(y)

        print(x)
        print(y)

        num_batches = y.shape[0]//batch_sz

        for i in tqdm(range(iters), "Training Network"):
            
            run_loss = 0
            run_acc = 0

            for b in range(num_batches):
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                p = self.forward(x[:,b*batch_sz:(b+1)*batch_sz,:])
                #self.accuracy(p, y[b*batch_sz:(b+1)*batch_sz, :])
                loss = self.loss(p, y[b*batch_sz:(b+1)*batch_sz, :])

                run_loss += loss.detach()/num_batches
                run_acc += self.accuracy(p, y[b*batch_sz:(b+1)*batch_sz, :])/num_batches

                loss.backward()
                self.optimizer.step()
    
            self.historical_loss.append(copy(run_loss))  
            self.historical_accuracy.append(run_acc) 

        x = 1-torch.sum(x.reshape(y.shape[0], -1), dim=1, keepdim=True)%2
        print(x.shape)
        print(self.accuracy(x, y))

    def accuracy(self, p, y):

        accuracy = ((p > 0.5) == (y > 0.5)).type(torch.FloatTensor).mean()

        return accuracy

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