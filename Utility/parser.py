"""
@Time: 2022/7/2 14:45
@Author: gorgeousdays@outlook.com
@File: parser.py
@Summary: NSDE argparse
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run NSDE.")
    parser.add_argument('--data_path', nargs='?', default='./Data/data.csv',
                        help='Input data path.')
    parser.add_argument('--weights_path', nargs='?', default='./Model/',
                        help='Store model path.')
    parser.add_argument('--train_rate', default=0.8,
                        help='The rate of train data of all dataset.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size.')
    parser.add_argument('--lr', default=1e-3,
                        help='Learning rate.')

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--m', type=int, default=244,
                        help='node nums')
    parser.add_argument('--n', type=int, default=1000,
                        help='Simulation times ')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=400,
                        help='Number of epoch.')

    return parser.parse_args()
