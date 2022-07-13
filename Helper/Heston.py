"""
@Time: 2022/7/2 9:22
@Author: gorgeousdays@outlook.com
@File: Heston.py
@Summary: Heston Model(Monte Carlo simulation)
"""

import numpy as np
from math import exp, sqrt
import numpy as np
import pandas as pd
from Utility.load_data import *
from Utility.helper import *


def Modify_Heston_Eurcall(S0, rf, sigma0, kappa, theta, delta, rho, T, K, N):
    """

    Args:
        S0: underlying assets price
        rf: risk free rate
        sigma0: the volatility of S(t)
        kappa:
        theta:
        delta: the volatility of V(t)
        rho: correlation coefficient
        T: expiry date
        K: strike price
        N: node nums

    Returns:
        P: premium
    """
    dt = T / N
    cor = np.zeros((2, 2))
    cor[0, 0], cor[1, 1] = 1, 1
    cor[0, 1], cor[1, 0] = rho, rho
    u = np.linalg.cholesky(cor)  # cor is positive definite matrix; np.dot(u,u.T) == cor
    S = np.zeros(N + 1)
    sigma = np.zeros(N + 1)

    S[0] = S0
    sigma[0] = sigma0
    for i in range(1, N + 1):
        z = np.random.randn(2, 1)

        # z1 = np.dot(u,z)
        z1 = np.zeros((2, 1))
        z1[0, 0] = z[0, 0] * u[0, 0] + z[1, 0] * u[0, 1]
        z1[1, 0] = z[0, 0] * u[1, 0] + z[1, 0] * u[1, 1]

        sigma[i] = sigma[i - 1] + kappa * (theta - sigma[i - 1]) * dt + delta * sqrt(dt) * z1[1, 0]
        # if sigma[i] < 0:
        #     print(sigma[i])
        # S[i] = S[i - 1] + rf * S[i - 1] * dt + sqrt(sigma[i]) * S[i-1] * sqrt(dt) * z1[0, 0]
        S[i] = S[i - 1] + rf * S[i - 1] * dt + sigma[i - 1] * S[i - 1] * sqrt(dt) * z1[0, 0] + 0.5 * (
                sigma[i - 1] ** 2) * S[i - 1] * dt * (z1[0, 0] ** 2 - 1)

    P = exp(-rf * T) * max(S[N] - K, 0)
    return P


def Heston_Eurcall(S0, rf, sigma0, kappa, theta, delta, rho, T, K, N):
    """

    Args:
        S0: underlying assets price
        rf: risk free rate
        sigma0: the volatility of S(t)
        kappa:
        theta:
        delta: the volatility of V(t)
        rho: correlation coefficient
        T: expiry date
        K: strike price
        N: node nums

    Returns:
        P: premium
    """
    dt = T / N
    cor = np.zeros((2, 2))
    cor[0, 0], cor[1, 1] = 1, 1
    cor[0, 1], cor[1, 0] = rho, rho
    u = np.linalg.cholesky(cor)  # cor is positive definite matrix; np.dot(u,u.T) == cor
    S = np.zeros(N + 1)
    sigma = np.zeros(N + 1)

    S[0] = S0
    sigma[0] = sigma0
    for i in range(1, N + 1):
        z = np.random.randn(2, 1)

        # z1 = np.dot(u,z)
        z1 = np.zeros((2, 1))
        z1[0, 0] = z[0, 0] * u[0, 0] + z[1, 0] * u[0, 1]
        z1[1, 0] = z[0, 0] * u[1, 0] + z[1, 0] * u[1, 1]

        if sigma[i - 1] < 0:
            sigma[i] = sigma[i - 1] + kappa * (theta - sigma[i - 1]) * dt - delta * sqrt(abs(sigma[i - 1])) * sqrt(dt) * \
                       z1[1, 0]
        else:
            sigma[i] = sigma[i - 1] + kappa * (theta - sigma[i - 1]) * dt + delta * sqrt(sigma[i - 1]) * sqrt(dt) * z1[
                1, 0]
        # S[i] = S[i - 1] + rf * S[i - 1] * dt + sqrt(sigma[i]) * S[i-1] * sqrt(dt) * z1[0, 0]
        S[i] = S[i - 1] + rf * S[i - 1] * dt + sigma[i - 1] * S[i - 1] * sqrt(dt) * z1[0, 0] + 0.5 * (
                sigma[i - 1] ** 2) * S[i - 1] * dt * (z1[0, 0] ** 2 - 1)

    P = exp(-rf * T) * max(S[N] - K, 0)
    return P


def generate_exp1_data():
    rf, kappa, theta, delta = 0.025, 1.5, 0.1, 0.3
    rho = -0.5
    N = 250
    S0, V0 = 100, 0.04
    times = 1000

    T_list_train = [1 / 12, 2 / 12, 3 / 12, 6 / 12, 1]
    K_list_train = [60, 70, 80, 90, 100, 110, 120, 130, 140]
    T_list_test = [1 / 12, 2 / 12, 3 / 12, 4 / 12, 5 / 12, 6 / 12, 8 / 12, 9 / 12, 10 / 12, 1]
    K_list_test = [60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140]

    df_train = pd.DataFrame(data=None)
    # Train Data
    for T in T_list_train:
        for K in K_list_train:
            P = np.zeros(times)
            for i in range(times):
                P[i] = Heston_Eurcall(S0, rf, V0, kappa, theta, delta, rho, T, K, N)
            P_pred = sum(P) / times
            if P_pred == 0:
                P_pred = 0.00001
            df_train = pd.concat([df_train, pd.DataFrame([[S0, T, K, P_pred]])])

    df_train.columns = ["S0", "T", "K", "P"]
    df_train.to_csv("../Data/exp1_train.csv", index=False)

    df_test = pd.DataFrame(data=None)
    for T in T_list_test:
        for K in K_list_test:
            P = np.zeros(times)
            for i in range(times):
                P[i] = Heston_Eurcall(S0, rf, V0, kappa, theta, delta, rho, T, K, N)
            P_pred = sum(P) / times
            df_test = pd.concat([df_test, pd.DataFrame([[S0, T, K, P_pred]])])

    df_test.columns = ["S0", "T", "K", "P"]
    df_test.to_csv("../Data/exp1_test.csv", index=False)


if __name__ == "__main__":
    # generate_exp1_data()

    _rf, _kappa, _theta, _delta = 0.025, 1.5, 0.1, 0.3
    _rho = -0.5
    _N = 250
    S0, V0 = 100, 0.04
    times = 100

    data_generator = Data(path="../Data/data.csv", train_rate=0.8)
    MAE = 0
    _T, _K = 1, 105
    T_all, K_all, P_all = [], [], []
    # for _S0, _T, _K, _P in zip(data_generator.S0_test, data_generator.T_test, data_generator.K_test,
    #                            data_generator.P_test):
    for _S0, _T, _K, _P in zip(data_generator.S0_train, data_generator.T_train, data_generator.K_train,
                               data_generator.P_train):

        P = np.zeros(times)
        for i in range(times):
            P[i] = Heston_Eurcall(_S0, _rf, V0, _kappa, _theta, _delta, _rho, _T, _K, _N)

        P_pred = sum(P) / times
        MAE += abs(P_pred - _P)

        T_all.append(_T)
        K_all.append(_K)
        P_all.append(P_pred)

    # print("MAE:", MAE / data_generator.n_test)
    print("MAE:", MAE / data_generator.n_train)

    T_all = np.array(T_all)
    K_all = np.array(K_all)
    P_all = np.array(P_all)
    # plot_3D(K_all, T_all, P_all, size=(17,10))
