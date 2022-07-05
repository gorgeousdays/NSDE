"""
@Time: 2022/7/4 20:31
@Author: gorgeousdays@outlook.com
@File: load_data.py
@Summary: DataGenerator
"""
import pandas as pd
import numpy as np


class Data(object):
    def __init__(self, path, train_rate):

        dataset = pd.read_csv(path)
        dataset = dataset.iloc[np.random.permutation(len(dataset))]  # shuffle

        self.S0 = np.array(dataset["S0"])
        self.T = np.array(dataset["T"])
        self.K = np.array(dataset["K"])
        self.P = np.array(dataset["P"])

        self.train_size = int(len(dataset) * train_rate)

        # split train data and test data
        self.S0_train, self.S0_test = self.S0[:self.train_size], self.S0[self.train_size:]
        self.T_train, self.T_test = self.T[:self.train_size], self.T[self.train_size:]
        self.K_train, self.K_test = self.K[:self.train_size], self.K[self.train_size:]
        self.P_train, self.P_test = self.P[:self.train_size], self.P[self.train_size:]
