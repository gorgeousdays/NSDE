"""
@Time: 2022/7/7 18:05
@Author: gorgeousdays@outlook.com
@File: BS.py
@Summary: Black-Scholes Model
"""
import math
import numpy as np
from scipy.stats import norm

from Utility.load_data import *
from Utility.helper import *

def BS_Eur(St, K, t, T, r, sigma, call=True):
    d1 = (np.log(St / K) + (r + 0.5 * sigma ** 2) * (T - t)) / (sigma * np.sqrt((T - t)))
    d2 = d1 - sigma * np.sqrt((T - t))
    if call:
        return St * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)
    else:
        return K * np.exp(-r * (T - t)) * norm.cdf(-d2) - St * norm.cdf(-d1)


if __name__ == "__main__":
    St, K, t, T, r, sigma = 100, 98, 0, 1, 0.03, 0.25

    data_generator = Data(path="../Data/data.csv", train_rate=0.8)

    MAE, MRE = 0., 0.
    T_all, K_all, P_all = [], [], []
    # for St, T, K, P in zip(data_generator.S0_test, data_generator.T_test, data_generator.K_test,
    #                        data_generator.P_test):
    for St, T, K, P in zip(data_generator.S0_train, data_generator.T_train, data_generator.K_train,
                           data_generator.P_train):
        P_pred = BS_Eur(St, K, t, T, r, sigma, call=True)

        MAE += np.abs(P_pred - P)
        MRE += np.abs((P_pred - P) / P)

        T_all.append(T)
        K_all.append(K)
        P_all.append(P)

    print("MAE:", MAE / data_generator.n_train)
    print("MRE:", MRE / data_generator.n_train)
    # print("MAE:", MAE / data_generator.n_test)
    # print("MRE:", MRE / data_generator.n_test)

    T_all = np.array(T_all)
    K_all = np.array(K_all)
    P_all = np.array(P_all)
    # plot_3D(K_all, T_all, P_all, size=(17, 10))
