"""
@Time: 2022/7/12 21:50
@Author: gorgeousdays@outlook.com
@File: ANN.py
@Summary: 
"""
import torch
import math
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from Utility.load_data import *
from time import time


class NN(torch.nn.Module):
    def __init__(self, layers):
        super(NN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        # self.activation = torch.nn.Tanh
        self.activation = torch.nn.ReLU

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def get_MAE_loss(self, P_pred, P):
        P = torch.tensor(P).float()
        return torch.sum((P_pred - P) ** 2)

    def forward(self, x):
        out = self.layers(x)
        # out = F.tanh(out)
        return out


if __name__ == "__main__":
    data_generator = Data(path="../Data/data.csv", train_rate=0.8)
    # layers = [4, 20, 20, 20, 20, 1]
    layers = [4, 20, 1]
    # batch_size越小 效果越好？
    batch_size = 1
    rf = 0.03

    ANN = NN(layers)

    optimizer = torch.optim.SGD(
        params=ANN.parameters(),
        lr=1e-3,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False)
    t0 = time()
    loss_logger, train_MAE_logger, test_MAE_logger = [], [], []

    for epoch in range(400):
        t1 = time()

        ANN.train()
        loss = 0.
        train_MAE = 0.
        n_batch = data_generator.n_train // batch_size + 1
        for idx in range(n_batch):
            S0, K, T, P = data_generator.sample(idx, batch_size)
            nrf = torch.full((len(S0), 1), rf).float()
            S0 = torch.tensor(S0.reshape(len(S0), 1)).float()
            K = torch.tensor(K.reshape(len(S0), 1)).float()
            T = torch.tensor(T.reshape(len(S0), 1)).float()
            input = torch.concat((S0, K, T, nrf), dim=1)
            P_pred = ANN(input)
            batch_loss = ANN.get_MAE_loss(P_pred, P)
            train_MAE += np.sum(np.abs(P_pred.cpu().detach().numpy() - P))

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

        train_MAE = train_MAE / data_generator.n_train
        if (epoch + 1) % 10 != 0:
            pref_str = 'Epoch %d [%.1fs]: loss==[%.5f], Train MAE=[%.5f]' % (epoch, time() - t1, loss, train_MAE)
            print(pref_str)
            continue

        # Test
        t2 = time()
        ANN.eval()
        with torch.no_grad():
            test_MAE = 0

            n_test_batch = data_generator.n_test // batch_size + 1
            for idx in range(n_test_batch):
                S0, K, T, P = data_generator.sample(idx, batch_size, isTrain=False)
                nrf = torch.full((len(S0), 1), rf).float()
                S0 = torch.tensor(S0.reshape(len(S0), 1)).float()
                K = torch.tensor(K.reshape(len(S0), 1)).float()
                T = torch.tensor(T.reshape(len(S0), 1)).float()
                input = torch.concat((S0, K, T, nrf), dim=1)
                P_pred = ANN(input)
                test_MAE += np.sum(np.abs(P_pred.cpu().detach().numpy() - P))

            test_MAE = test_MAE / data_generator.n_test
        t3 = time()


        loss_logger.append(loss)
        train_MAE_logger.append(train_MAE)
        test_MAE_logger.append(test_MAE)

        perf_str = 'Epoch %d [%.1fs + %.1fs]: loss==[%.5f], Train MAE=[%.5f],  Test MAE=[%.5f], ' % (
            epoch, t2 - t1, t3 - t2, loss, train_MAE, test_MAE)
        print(perf_str)