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

        dataset = dataset[:50]

        self.S0 = np.array(dataset["S0"])
        self.T = np.array(dataset["T"])
        self.T = self.T / 365
        self.K = np.array(dataset["K"])
        self.P = np.array(dataset["P"])

        self.n_train = int(len(dataset) * train_rate)
        self.n_test = len(dataset) - self.n_train

        # Exp1

        # dataset_train = pd.read_csv('../Data/exp1_train.csv')
        # dataset_test = pd.read_csv('../Data/exp1_test.csv')
        # self.n_train, self.n_test = len(dataset_train), len(dataset_test)
        # dataset = pd.concat([dataset_train, dataset_test])
        # self.S0 = np.array(dataset["S0"])
        # self.T = np.array(dataset["T"])
        # self.K = np.array(dataset["K"])
        # self.P = np.array(dataset["P"])

        # End Exp1

        # split train data and test data
        self.S0_train, self.S0_test = self.S0[:self.n_train], self.S0[self.n_train:]
        self.T_train, self.T_test = self.T[:self.n_train], self.T[self.n_train:]
        self.K_train, self.K_test = self.K[:self.n_train], self.K[self.n_train:]
        self.P_train, self.P_test = self.P[:self.n_train], self.P[self.n_train:]

    def sample(self, idx, batch_size, isTrain=True):
        if isTrain:
            if (idx + 1) * batch_size < self.n_train:
                return self.S0_train[idx * batch_size: (idx + 1) * batch_size], \
                        self.K_train[idx * batch_size: (idx + 1) * batch_size], \
                        self.T_train[idx * batch_size: (idx + 1) * batch_size], \
                        self.P_train[idx * batch_size: (idx + 1) * batch_size]
            else:
                return self.S0_train[idx * batch_size:], \
                        self.K_train[idx * batch_size:], \
                        self.T_train[idx * batch_size:], \
                        self.P_train[idx * batch_size:]
        else:
            if (idx + 1) * batch_size < self.n_test:
                return self.S0_test[idx * batch_size: (idx + 1) * batch_size], \
                        self.K_test[idx * batch_size: (idx + 1) * batch_size], \
                        self.T_test[idx * batch_size: (idx + 1) * batch_size], \
                        self.P_test[idx * batch_size: (idx + 1) * batch_size]
            else:
                return self.S0_test[idx * batch_size:], \
                        self.K_test[idx * batch_size:], \
                        self.T_test[idx * batch_size:], \
                        self.P_test[idx * batch_size:]
