#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import numpy as np
from taurus import operations


class Pooling(operations.Operation):

    def __init__(self):
        super(Pooling, self).__init__()

    def forward_cpu(self, feature):
        pass

    def backprop_cpu(self, pool_out_delta, pool_out_max_location):
        pass

    def forward_gpu(self):
        pass

    def backprop_gpu(self):
        pass


class Pooling2D(Pooling):

    def __init__(self):
        super(Pooling2D, self).__init__()


class MaxPooling2D(Pooling2D):

    size = None
    stride = None

    def __init__(self, size=2, stride=2):
        super(MaxPooling2D, self).__init__()

        self.size = size
        self.stride = stride

    def __call__(self, inputs, *args, **kwargs):

        feature = inputs

        pool_out, pool_out_max_location = self._forward_cpu(feature)

        return pool_out, pool_out_max_location

    def backprop(self, pool_out_delta, pool_out_max_location):

        # 目前默认cpu
        delta = self._backprop_cpu(pool_out_delta, pool_out_max_location)

        return delta

    def _forward_cpu(self, feature):

        """最大池化操作
        同时输出池化后的结果以及用于记录最大位置的张量，方便之后delta误差反向传播"""
        pool_out = np.zeros([np.uint16((feature.shape[0] - self.size) / self.stride + 1),
                             np.uint16((feature.shape[1] - self.size) / self.stride + 1),
                             feature.shape[2]])

        pool_out_max_location = np.zeros(pool_out.shape)  # 该函数用于记录最大值位置

        for ch_num in range(feature.shape[-1]):
            r_out = 0
            for r in np.arange(0, feature.shape[0] - self.size + 1, self.stride):
                c_out = 0
                for c in np.arange(0, feature.shape[1] - self.size + 1, self.stride):
                    # 取最大值
                    pool_out[r_out, c_out, ch_num] = np.max(feature[r:r + self.size, c:c + self.size, ch_num])
                    # 记录最大点位置
                    pool_out_max_location[r_out, c_out, ch_num] = np.argmax(feature[r:r + self.size, c:c + self.size, ch_num])
                    c_out += 1
                r_out += 1

        return pool_out, pool_out_max_location

    def _backprop_cpu(self, pool_out_delta, pool_out_max_location):

        delta = np.zeros([np.uint16((pool_out_delta.shape[0] - 1) * self.stride + self.size),
                          np.uint16((pool_out_delta.shape[1] - 1) * self.stride + self.size),
                          pool_out_delta.shape[2]])

        for ch_num in range(pool_out_delta.shape[-1]):
            for r in range(pool_out_delta.shape[0]):
                for c in range(pool_out_delta.shape[1]):
                    order = pool_out_max_location[r, c, ch_num]
                    m = np.uint16(self.stride * r + order // self.size)
                    n = np.uint16(self.stride * c + order % self.size)
                    delta[m, n, ch_num] = pool_out_delta[r, c, ch_num]

        return delta
