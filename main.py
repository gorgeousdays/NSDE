"""
@Time: 2022/7/2 9:21
@Author: gorgeousdays@outlook.com
@File: main.py
@Summary: PyTorch Implementation for NSDE(OPTION PRICING BY NEURAL STOCHASTIC DIFFERENTIAL EQUATIONS:
            A SIMULATION-OPTIMIZATION APPROACH)
"""
import torch
import numpy as np
from time import time

from Utility.load_data import *
from Utility.parser import parse_args
from Utility.helper import *
from NSDE import NSDE
import warnings

warnings.filterwarnings('ignore')


def set_random_seed(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    print("SEED:", SEED)


def Initialization():
    # TODO: calibrate the initial values by the closed-form pricing formula of Heston Model
    V0 = 0.5
    rho = 0.5
    return V0, rho


args = parse_args()
data_generator = Data(path=args.data_path, train_rate=args.train_rate)

if __name__ == "__main__":

    set_random_seed(args.seed)

    args.device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu:0")

    m, n = args.m, args.n
    V0, rho = Initialization()

    # Layers For NN1-4
    layers = [
        [4, 20, 20, 20, 20, 20, 20, 20, 20, 1],
        [4, 10, 20, 20, 20, 20, 20, 20, 20, 1],
        [4, 20, 20, 20, 20, 20, 20, 20, 20, 1],
        [4, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    ]

    rf = 0.12

    model = NSDE(m, n, V0, rho, layers, args)

    t0 = time()

    loss_logger, MAE_logger = [], []
    for epoch in range(args.epoch):
        t1 = time()

        model.train()
        loss = 0
        for S0, K, T, P in zip(data_generator.S0_train, data_generator.K_train,
                               data_generator.T_train, data_generator.P_train):
            P_pred = model(S0, K, T, rf, P)
            loss = loss + torch.mean((P_pred - P) ** 2)
            # Find User Index
            # for index,data in enumerate(data_generator.S0_train):
            #     if data ==S0:
            #         print(index)
            #         break

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                pref_str = 'Epoch %d [%.1fs]: loss==[%.5f]' % (epoch, time() - t1, loss)
                print(pref_str)
            continue

        # Test
        t2 = time()
        model.eval()
        with torch.no_grad():
            MAE = 0
            for S0, K, T, P in zip(data_generator.S0_test, data_generator.K_test,
                                   data_generator.T_test, data_generator.P_test):
                P_pred = model(S0, K, T, rf, P)
                MAE += torch.mean((P_pred - P) ** 2)
        t3 = time()

        loss_logger.append(loss)
        MAE_logger.append(MAE)

        perf_str = 'Epoch %d [%.1fs + %.1fs]: loss==[%.5f], MAE=[%.5f], ' % (epoch, t2 - t1, t3 - t2, loss, MAE)
        print(perf_str)

        if args.save_flag == 1:
            savePath = args.weights_path + '/' + str(epoch) + '.pkl'
            ensureDir(savePath)
            torch.save(model.state_dict(), savePath)
            print('save the weights in path: ', savePath)