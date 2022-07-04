"""
@Time: 2022/7/2 14:45
@Author: gorgeousdays@outlook.com
@File: parser.py
@Summary: NSDE argparse
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run NSDE.")

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--epoch', type=int, default=400,
                        help='Number of epoch.')

    return parser.parse_args()
