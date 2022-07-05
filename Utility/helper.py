"""
@Time: 2022/7/5 18:34
@Author: gorgeousdays@outlook.com
@File: helper.py
@Summary: 
"""

import os

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)