"""
@Time: 2022/7/2 15:01
@Author: gorgeousdays@outlook.com
@File: Heston.py
@Summary: DNN model and Heston model
"""
import torch
import math
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F


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

    def forward(self, x):
        out = self.layers(x)
        out = F.tanh(out)
        # out = out * 0.1
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
        self.device = args.device

        self.m = m
        self.n = n
        self.V0 = torch.tensor(V0).to(args.device)
        self.V0.requires_grad = True
        self.rho = torch.tensor(rho).to(args.device)
        self.rho.requires_grad = True

        self.layers = layers

        self.NN1 = NN(layers[0]).to(self.device)
        self.NN2 = NN(layers[1]).to(self.device)
        self.NN3 = NN(layers[2]).to(self.device)
        self.NN4 = NN(layers[3]).to(self.device)

    def get_MAE_loss(self, P_pred, P):
        P = torch.tensor(P).float().to(self.device)
        return torch.sum((P_pred - P) ** 2)

    def forward(self, S0, K, T, rf):
        """

        Args:
            S0: the price of the underlying asset at time 0
            K: strike price
            T: expiry date
            rf: risk-free interest rate
        Returns:

        """
        data_size = len(S0)

        dt = T / self.m
        # Shape: 1000 * batch_size * 1
        ndt = torch.tensor(dt.reshape(data_size, 1)).unsqueeze(0).repeat(self.n, 1, 1).float().to(self.device)
        nrf = torch.full_like(ndt, rf).float().to(self.device)

        S = torch.tensor(S0.reshape(data_size, 1)).unsqueeze(0).repeat(self.n, 1, 1).float().to(self.device)
        V = self.V0.repeat(S.shape[0] * S.shape[1]).reshape_as(S)

        K = torch.tensor(K.reshape(1, data_size)).repeat(self.n, 1).float().to(self.device)

        T = torch.tensor(T).float().to(self.device)
        for j in range(0, self.m):

            Z1 = torch.tensor(np.random.standard_normal(size=(self.n, data_size))).float().to(self.device)
            Z2 = self.rho * Z1 + math.sqrt(1 - self.rho ** 2) * \
                 torch.tensor(np.random.standard_normal(size=(self.n, data_size))).float().to(self.device)

            N1 = self.NN1(torch.cat([S, V, nrf, ndt * j], dim=2))
            N2 = self.NN2(torch.cat([S, V, nrf, ndt * j], dim=2))
            N3 = self.NN3(torch.cat([S, V, nrf, ndt * j], dim=2))
            N4 = self.NN4(torch.cat([S, V, nrf, ndt * j], dim=2))

            V = V * (1 + N3 * ndt + N4 * torch.sqrt(ndt) * Z2.reshape(self.n, len(S0), 1))
            S = S * (1 + N1 * ndt + N2 * torch.sqrt(ndt) * Z1.reshape(self.n, len(S0), 1))

            # Ablation Study
            # S = S * (1 + nrf * ndt + N2 * torch.sqrt(ndt) * Z1.reshape(self.n, len(S0), 1))

        S = S.squeeze()
        zero_tensor = torch.zeros((self.n, len(S0))).to(self.device)
        P_future = torch.where(S - K < 0, zero_tensor, S - K)
        P_pred = torch.sum(torch.exp(-rf * T) * P_future, axis=0) / self.n

        return P_pred

    def forward_for_one(self, S0, K, T, rf):
        """

        Args:
            S0: the price of the underlying asset at time 0
            K: strike price
            T: expiry date
            rf: risk-free interest rate
        Returns:

        """
        Z1 = torch.tensor(np.random.standard_normal(size=(self.n, self.m))).float().to(self.device)
        Z2 = self.rho * Z1 + math.sqrt(1 - self.rho ** 2) * \
             torch.tensor(np.random.standard_normal(size=(self.n, self.m))).float().to(self.device)

        dt = T / self.m

        ndt = torch.tensor(dt).expand(self.n, 1).reshape(self.n, 1).float().to(self.device)
        nrf = torch.tensor(rf).expand(self.n, 1).reshape(self.n, 1).float().to(self.device)

        S = torch.full([self.n, 1], float(S0))
        V = torch.full([self.n, 1], float(self.V0))

        for j in range(0, self.m):
            S = S * (1 +
                     self.NN1(torch.cat([S, V, nrf, ndt * j], dim=1)) * ndt +
                     self.NN2(torch.cat([S, V, nrf, ndt * j], dim=1)) * torch.sqrt(ndt) * Z1[:, j].reshape(self.n, 1)
                     )
            V = V * (1 +
                     self.NN3(torch.cat([S, V, nrf, ndt * j], dim=1)) * ndt +
                     self.NN4(torch.cat([S, V, nrf, ndt * j], dim=1)) * torch.sqrt(ndt) * Z2[:, j].reshape(self.n,
                                                                                                           1)
                     )

        zero_tensor = torch.zeros((self.n, 1))
        P_future = torch.where(S - torch.tensor(K) < 0, zero_tensor, S)
        P_pred = torch.sum(math.exp(-rf * T) * P_future) / self.n

        return P_pred
