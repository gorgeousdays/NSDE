"""
@Time: 2022/7/2 9:21
@Author: gorgeousdays@outlook.com
@File: main.py
@Summary: PyTorch Implementation for NSDE(OPTION PRICING BY NEURAL STOCHASTIC DIFFERENTIAL EQUATIONS:
            A SIMULATION-OPTIMIZATION APPROACH)
"""
import torch
import itertools
import numpy as np

from Utility.parser import parse_args
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
    V0 = 1
    rho = 1
    return V0, rho


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)

    args.device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu:0")

    # TODO: load data
    # S_train, T_train, V_train, S_test, T_test, V_test = load_data()
    S0 = 20

    # Layers For NN1-4
    layers = [
        [4, 20, 20, 20, 20, 20, 20, 20, 20, 1],
        [4, 10, 20, 20, 20, 20, 20, 20, 20, 1],
        [4, 20, 20, 20, 20, 20, 20, 20, 20, 1],
        [4, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    ]

    T, m, n, rf, K, P = 1, 244, 1000, 0.12, 22, 1
    D = args.epoch

    V0, rho = Initialization()
    model = NSDE(m, n, V0, rho, layers, args)

    model.train()
    model.optimizer.zero_grad()
    P_pred = model(S0, K, T, rf, P)
    loss = torch.mean((P_pred - P) ** 2)
    loss.backward()

    model.optimizer.step()

    model.eval()
    with torch.no_grad():
        V_pred = model(100, 22, 1, 0.12, 23)

    error = np.linalg.norm(V_test - V_pred, 2) / np.linalg.norm(V_test, 2)
    print("Error :%e" % error)
