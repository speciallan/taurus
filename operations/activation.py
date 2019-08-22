#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import numpy as np
import time
from taurus import operations


def sigmoid(z):
    a = 1.0 / (1.0 + np.exp(-z))
    return a


def sigmoid_prime(z):
    """sigmoid函数的一阶导数"""
    return sigmoid(z) * (1 - sigmoid(z))


class Activation(operations.Operation):

    def __init__(self):
        super(Activation, self).__init__()

    def __call__(self, inputs, *args, **kwargs):
        pass


class Sigmoid(operations.Operation):

    def __init__(self):
        super(Sigmoid, self).__init__()

    def __call__(self, inputs, *args, **kwargs):

        x = inputs[0]
        z = 1.0 / (1.0 + np.exp(-x))
        return z


class Relu(Activation):

    def __init__(self):
        super(Relu, self).__init__()

    def __call__(self, feature, *args, **kwargs):
        pass


def relu(feature, version=0):
    '''Relu激活函数，有两种情况会使用到
    当在卷积层中使用时，feature为一个三维张量，，[行，列，通道]
    当在全连接层中使用时，feature为一个列向量
    0: 1.8xe-5
    1: 0.004
    '''
    # time1 = time.time()

    if version == 0:
        relu_out =  np.where(feature <= 0, 0, feature)
    else:
        relu_out = np.zeros(feature.shape)

        if len(feature.shape) > 2:
            for ch_num in range(feature.shape[-1]):
                for r in range(feature.shape[0]):
                    for c in range(feature.shape[1]):
                        relu_out[r, c, ch_num] = max(feature[r, c, ch_num], 0)
        else:
            for r in range(feature.shape[0]):
                relu_out[r, 0] = max(feature[r, 0], 0)

    # print('relu:{}'.format(time.time() - time1))

    return relu_out

def relu_prime(feature, version=0):  # 对relu函数的求导
    '''relu函数的一阶导数，间断点导数认为是0
    0: 9xe-6
    1: 4xe-5
    '''
    # time1 = time.time()

    if version == 0:
        relu_prime_out = np.where(feature <= 0, 0, 1)
    else:
        relu_prime_out = np.zeros(feature.shape)

        if len(feature.shape) > 2:
            for ch_num in range(feature.shape[-1]):
                for r in range(feature.shape[0]):
                    for c in range(feature.shape[1]):
                        if feature[r, c, ch_num] > 0:
                            relu_prime_out[r, c, ch_num] = 1
        else:
            for r in range(feature.shape[0]):
                if feature[r, 0] > 0:
                    relu_prime_out[r, 0] = 1

    # print('relu_prime:{}'.format(time.time() - time1))

    return relu_prime_out


def softmax(z):

    tmp = np.max(z)
    z -= tmp  # 用于缩放每行的元素，避免溢出，有效
    z = np.exp(z)
    tmp = np.sum(z)
    z /= tmp
    return z
