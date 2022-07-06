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
set_random_seed(args.seed)
data_generator = Data(path=args.data_path, train_rate=args.train_rate)

if __name__ == "__main__":

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

    cur_best_pre_0, stopping_step = 0, 0
    loss_logger, train_MAE_logger, test_MAE_logger = [], [], []

    for epoch in range(args.epoch):
        t1 = time()

        model.train()
        loss = 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        for idx in range(n_batch):
            S0, K, T, P = data_generator.sample(idx, args.batch_size)

            P_pred = model(S0, K, T, rf)
            batch_loss = model.get_MAE_loss(P_pred, P)

            model.optimizer.zero_grad()
            batch_loss.backward()
            model.optimizer.step()

            loss += batch_loss
        train_MAE = loss / data_generator.n_train

        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                pref_str = 'Epoch %d [%.1fs]: loss==[%.5f], Train MAE=[%.5f]' % (epoch, time() - t1, loss, train_MAE)
                print(pref_str)
            continue

        # Test
        t2 = time()
        model.eval()
        with torch.no_grad():
            test_MAE = 0

            n_test_batch = data_generator.n_test // args.batch_size + 1
            for idx in range(n_test_batch):
                S0, K, T, P = data_generator.sample(idx, args.batch_size, isTrain=False)

                P_pred = model(S0, K, T, rf)
                test_MAE += torch.sum((P_pred - P) ** 2)

            test_MAE = test_MAE / data_generator.n_test
        t3 = time()

        loss_logger.append(loss)
        train_MAE_logger.append(train_MAE)
        test_MAE_logger.append(test_MAE)

        perf_str = 'Epoch %d [%.1fs + %.1fs]: loss==[%.5f], Train MAE=[%.5f],  Test MAE=[%.5f], ' % (
            epoch, t2 - t1, t3 - t2, loss, train_MAE, test_MAE)
        print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(test_MAE, cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=15)

        if should_stop:
            break

        if args.save_flag == 1:
            savePath = args.weights_path + '/' + str(epoch) + '.pkl'
            ensureDir(savePath)
            torch.save(model.state_dict(), savePath)
            print('save the weights in path: ', savePath)

    MAES = np.array(test_MAE_logger)
    best_rec_0 = max(MAES[:, 0])
    idx = list(MAES[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\t test_MAE=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in MAES[idx]]))
    print(final_perf)