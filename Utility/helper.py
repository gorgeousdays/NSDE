"""
@Time: 2022/7/5 18:34
@Author: gorgeousdays@outlook.com
@File: helper.py
@Summary: 
"""

import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (
            expected_order == 'dec' and log_value <= best_value) or best_value == 0:
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


def plot_3D(x, y, z, size):
    X = x.reshape(size)
    Y = y.reshape(size)
    Z = z.reshape(size)

    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

    ax.set_title('Option Price')
    ax.set_xlabel('Strike price')
    ax.set_ylabel('Time to maturity')
    ax.set_zlabel('')
    plt.show()
