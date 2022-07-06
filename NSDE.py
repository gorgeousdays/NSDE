"""
@Time: 2022/7/2 15:01
@Author: gorgeousdays@outlook.com
@File: Heston.py
@Summary: DNN model and Heston model
"""
import torch
import math
import numpy as np
import itertools
from collections import OrderedDict


class NN(torch.nn.Module):
    def __init__(self, layers):
        super(NN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh

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

    def forward(self, x):
        out = self.layers(x)
        return out


class NSDE(torch.nn.Module):
    def __init__(self, m, n, V0, rho, layers, args):
        super(NSDE, self).__init__()
        """

        Args:
            m: node nums
            n: simulation times
            V0: the volatility at time 0
            rho: correlation coefficient
            layers: the layers for NN
            args: other parameters
        """
        self.m = m
        self.n = n
        self.V0 = V0
        self.rho = rho

        self.layers = layers

        self.device = args.device

        self.NN1 = NN(layers[0]).to(self.device)
        self.NN2 = NN(layers[1]).to(self.device)
        self.NN3 = NN(layers[2]).to(self.device)
        self.NN4 = NN(layers[3]).to(self.device)

        self.optimizer = torch.optim.SGD(
            params=itertools.chain(self.NN1.parameters(),
                                   self.NN2.parameters(),
                                   self.NN3.parameters(),
                                   self.NN4.parameters()),
            lr=args.lr,
            momentum=0,
            dampening=0,
            weight_decay=0,
            nesterov=False)

        self.Z1 = torch.tensor(np.random.normal(size=(n, m))).float().to(self.device)
        self.Z2 = (rho * self.Z1 + math.sqrt(1 - rho ** 2) * np.random.normal(size=(n, m))).float().to(self.device)

    def get_MAE_loss(self, P_pred, P):
        # P_pred = torch.tensor(P_pred).float().to(self.device)
        P = torch.tensor(P).float().to(self.device)

        return torch.sum((P_pred - P) ** 2)

    def forward(self, S0, K, T, rf):
        dt = T / self.m

        ndt = torch.tensor(dt.reshape(len(dt), 1)).unsqueeze(0).repeat(self.n, 1, 1).float().to(self.device)
        nrf = torch.full_like(ndt, rf).float().to(self.device)

        S = torch.tensor(S0.reshape(len(S0), 1)).unsqueeze(0).repeat(self.n, 1, 1).float().to(self.device)
        V = torch.full_like(S, self.V0).float().to(self.device)
        K = torch.tensor(K.reshape(1,len(K))).repeat(self.n, 1).float().to(self.device)
        for j in range(0, self.m):
            S = S * (1 +
                     self.NN1(torch.cat([S, V, nrf, ndt * j], dim=2)) * ndt +
                     self.NN2(torch.cat([S, V, nrf, ndt * j], dim=2)) *
                     self.Z1[:, j].reshape(self.n, 1).unsqueeze(1).repeat(1, len(S0), 1)
                     )
            V = V * (1 +
                     self.NN3(torch.cat([S, V, nrf, ndt * j], dim=2)) * ndt +
                     self.NN4(torch.cat([S, V, nrf, ndt * j], dim=2)) * torch.sqrt(ndt) *
                     self.Z2[:, j].reshape(self.n, 1).unsqueeze(1).repeat(1, len(S0), 1)
                     )
            break

        S = S.squeeze()
        zero_tensor = torch.zeros((self.n, len(S0)))
        P_future = torch.where(S - K < 0, zero_tensor, S)
        P_pred = torch.sum(torch.exp(-rf * torch.tensor(T)) * P_future, axis=0) / self.n

        return P_pred



    def forward_for_one_line(self, S0, K, T, rf, P):
        """

        Args:
            S0: the price of the underlying asset at time 0
            K: strike price
            T: expiry date
            rf: risk-free interest rate
            P: strike price
        Returns:

        """

        dt = T / self.m

        ndt = torch.tensor(dt).expand(self.n, 1).reshape(self.n, 1).float().to(self.device)
        nrf = torch.tensor(rf).expand(self.n, 1).reshape(self.n, 1).float().to(self.device)

        S = torch.full([self.n, 1], S0)
        V = torch.full([self.n, 1], self.V0)

        for j in range(0, self.m):
            S = S * (1 +
                     self.NN1(torch.cat([S, V, nrf, ndt * j], dim=1)) * ndt +
                     self.NN2(torch.cat([S, V, nrf, ndt * j], dim=1)) * self.Z1[:, j].reshape(self.n, 1)
                     )
            V = V * (1 +
                     self.NN3(torch.cat([S, V, nrf, ndt * j], dim=1)) * ndt +
                     self.NN4(torch.cat([S, V, nrf, ndt * j], dim=1)) * torch.sqrt(ndt) * self.Z2[:, j].reshape(self.n,
                                                                                                                1)
                     )

        zero_tensor = torch.zeros((self.n, 1))
        P_future = torch.where(S - K < 0, zero_tensor, S)
        P_pred = torch.sum(math.exp(-rf * T) * P_future) / self.n

        return P_pred
