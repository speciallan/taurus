#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
sys.path.append('..')
import time

import numpy as np
import cupy as cp

def cupy():
    """
    <CUDA Device 0>
(10000, 10000) 2.967323064804077
(10000, 10000) 0.13281846046447754
(py36tf) ➜  taurus python test.py
    <CUDA Device 0>
(20000, 20000) 40.27280116081238
(20000, 20000) 0.19600987434387207
    <CUDA Device 0>
(30000, 30000) 143.53079319000244
(30000, 30000) 0.13195157051086426
    :return:
    """
    print(cp.cuda.Device())
    cp.cuda.Device(0).use()
    # x_cpu = np.array([1, 2, 3])
    # x_gpu = cp.array([1, 2, 3])

    x_cpu = np.random.randn(30000, 30000).astype(np.float32)
    x_gpu = cp.asarray(x_cpu)

    time1 = time.time()
    t1 = np.dot(x_cpu, x_cpu)
    print(t1.shape, time.time() - time1)

    time2 = time.time()
    t2 = cp.dot(x_gpu, x_gpu)
    print(t2.shape, time.time() - time2)
    # print(x_cpu, x_gpu)

    x_cpu = cp.asnumpy(x_gpu)


from keras.layers import *
from taurus.operations import *

def test():

    x = np.array([1,2,3])
    x = FC(nums=10)(x)
    print(x)


if __name__ == '__main__':
    # cupy()
    test()
