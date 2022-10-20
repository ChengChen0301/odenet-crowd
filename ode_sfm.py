import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--viz', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument("--mode", default='client')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

left = -5.0
right = 5.0
up = 5.0
down = -5.0
width = 1.0
door = np.array([right, 0])
wall = np.array([[right, up, left, up],  # 上
                 [left, up, left, down],  # 左
                 [left, down, right, down],  # 下
                 [right, down, door[0], -width],
                 [door[0], width, right, up]])

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)

Np = 5
Nknn = 4
Nsamples = 200
saveDir = "result/sfm"


class Dataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()
        self.nfe = 0

        self.net1 = nn.Sequential(
            nn.Linear(4, 2),
        )

        self.net2 = nn.Sequential(
            nn.Linear(1, 1),
        )

        self.net3 = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        for m in self.net1.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        for m in self.net2.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        for m in self.net3.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        self.nfe += 1
        y = y.float()
        theX = y[:, 0:2]
        theV = y[:, 2:4]
        ux = y[:, 4:2*Np+2]
        dxdt = theV

        unit_e = 1.0*(torch.tensor([right, 0]).to(device) - theX) * (1.0 / torch.sqrt(torch.sum(torch.square(
            theX - torch.tensor([right, 0]).to(device)), dim=1)).reshape(-1, 1))
        f_1 = self.net1(torch.cat([unit_e, theV], 1))

        rightdist = right - theX[:, 0].reshape(-1, 1)
        todoor = np.where((theX[:, 1].cpu() < width) & (theX[:, 1].cpu() > -width))[0]
        rightdist[todoor] = 100
        f_2 = torch.exp(self.net2(theX[:, 0].reshape(-1, 1) - left)) * torch.tensor([[1, 0]]).to(device) \
              + torch.exp(self.net2(theX[:, 1].reshape(-1, 1) - down)) * torch.tensor([[0, 1]]).to(device) \
              + torch.exp(self.net2(up - theX[:, 1].reshape(-1, 1))) * torch.tensor([[0, -1]]).to(device) \
              + torch.exp(self.net2(rightdist)) * torch.tensor([[-1, 0]]).to(device)

        f_3 = torch.zeros((len(y), 2)).to(device)
        for j in range(Nknn):
            d_ij = torch.sqrt(torch.sum(torch.square(ux[:, 2 * j:2 * j + 2]), dim=1))
            f_3 += self.net3(d_ij.reshape(-1, 1))*(-ux[:, 2*j:2*j+2]/d_ij.reshape(-1, 1))

        dvdt = (f_1 + f_2 + f_3) / 80.0
        dudt = torch.zeros((len(y), 2*Nknn)).to(device)
        result = torch.cat([dxdt, dvdt], 1)
        result = torch.cat([result, dudt], 1)
        return result


if __name__ == "__main__":

    saveDir = "result/sfm"
    makedirs(saveDir)

    train_x = torch.load(saveDir + '/train_x.pt')
    train_y = torch.load(saveDir + '/train_y.pt')
    test_x = torch.load(saveDir + '/test_x.pt')
    test_y = torch.load(saveDir + '/test_y.pt')
    train_dataset = Dataset(train_x, train_y)
    test_dataset = Dataset(test_x, test_y)

    minLoss = 100.0
    ii = 0

    func = ODEFunc().to(device)
    optimizer = optim.RMSprop(func.parameters(), lr=args.lr)
    batch_t = torch.tensor([0, 0.01]).to(device)
    lambda_regularizer = 0.01

    with open(saveDir + '/modelInfo.txt', mode='a') as filename:
        filename.write('the networks approximate force\n')
        filename.write('net1: 4-2\n')
        filename.write('net2: 1-1\n')
        filename.write('net3: 1-16-8-1\n')

    # train the network
    for epoch in range(1000):
        train_set = DataLoader(train_dataset, batch_size=600, shuffle=True)
        test_set = DataLoader(test_dataset, batch_size=300, shuffle=True)
        train_loss = 0.0
        for i, data in enumerate(train_set):
            optimizer.zero_grad()
            batch_y0, batch_y = data
            batch_y0 = batch_y0.to(device)
            batch_y = batch_y.to(device)
            pred_y = odeint(func, batch_y0, batch_t).to(device)
            loss = torch.mean(torch.abs(pred_y[1][:, 0:4] - batch_y))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / (i+1)

        with torch.no_grad():
            test_loss = 0.0
            for j, data in enumerate(test_set):
                batch_y0, batch_y = data
                batch_y0 = batch_y0.to(device)
                batch_y = batch_y.to(device)
                pred_y = odeint(func, batch_y0, batch_t).to(device)
                loss = torch.mean(torch.abs(pred_y[1][:, 0:4] - batch_y))
                test_loss += loss.item()
            test_loss = test_loss / (j+1)
            with open(saveDir + '/loss.txt', mode='a') as filename:
                filename.write('Epoch {:03d} | TrainLoss {:.6f} | TestLoss {:.6f}'.format(epoch, train_loss, test_loss))
                filename.write('\n')
            totalLoss = train_loss + test_loss
            if totalLoss < minLoss:
                minLoss = totalLoss
                minIndex = epoch
                torch.save(func.net1.state_dict(), saveDir + '/net1_parameter.pkl')
                torch.save(func.net2.state_dict(), saveDir + '/net2_parameter.pkl')
                torch.save(func.net3.state_dict(), saveDir + '/net3_parameter.pkl')

    with open(saveDir + '/loss.txt', mode='a') as filename:
        filename.write('minIndex {:03d} | minLoss {:.6f}'.format(minIndex, minLoss))
        filename.write('\n')
