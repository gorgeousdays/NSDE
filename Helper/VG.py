"""
@Time: 2022/7/9 10:37
@Author: gorgeousdays@outlook.com
@File: VG.py
@Summary:  Variance Gamma Model
"""
import numpy as np
import pandas as pd
import math


def VG_Eurcall(rf, sigma, theta, v, T, K, S0):
    up = (math.sqrt((theta ** 2 + 2 * sigma ** 2) / v) + theta) / 2
    vp = up ** 2 + v
    un = (math.sqrt((theta ** 2 + 2 * sigma ** 2) / v) - theta) / 2
    vn = un ** 2 - v
    Xt = math.pow(rf, up) - math.pow(rf, un)

    w = math.log(1 - theta * v - sigma ** 2 * v / 2) / v
    St = S0 * math.exp((rf + w) * T + Xt)

    P = math.exp(-rf * T) * max(St - K, 0)

    return P


def generate_exp2_data():
    rf = 0.025
    theta, v, sigma = -0.1436, 0.1686, 0.1231
    S0 = 100
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
                P[i] = VG_Eurcall(rf, sigma, theta, v, T, S0, K)
            P_pred = sum(P) / times
            df_train = pd.concat([df_train, pd.DataFrame([[S0, T, K, P_pred]])])

    df_train.columns = ["S0", "T", "K", "P"]
    df_train.to_csv("../Data/exp2_train.csv", index=False)

    df_test = pd.DataFrame(data=None)
    for T in T_list_test:
        for K in K_list_test:
            P = np.zeros(times)
            for i in range(times):
                P[i] = VG_Eurcall(rf, sigma, theta, v, T, S0, K)
            P_pred = sum(P) / times
            df_test = pd.concat([df_test, pd.DataFrame([[S0, T, K, P_pred]])])

    df_test.columns = ["S0", "T", "K", "P"]
    df_test.to_csv("../Data/exp2_test.csv", index=False)


if __name__ == "__main__":
    generate_exp2_data()

    # rf = 0.025
    # theta, v, sigma = -0.1436, 0.1686, 0.1231
    # S0 = 100
    # T = 1
    # K = 110
    # P = VG_Eurcall(rf, sigma, theta, v, T, S0, K)
