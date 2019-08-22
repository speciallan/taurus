#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import numpy as np
import time
from taurus import operations
from taurus.operations.common import im2col, col2im
from taurus.utils.spe import spe


class Pooling(operations.Operation):

    def __init__(self):
        super(Pooling, self).__init__()

    def forward_cpu(self, feature):
        pass

    def backprop_cpu(self, pool_out_delta):
        pass

    def forward_gpu(self):
        pass

    def backprop_gpu(self):
        pass


class Pooling2D(Pooling):

    def __init__(self):
        super(Pooling2D, self).__init__()


class MaxPooling2D(Pooling2D):

    def __init__(self, size=2, stride=2, pad=0):
        super(MaxPooling2D, self).__init__()

        self.size = size
        self.stride = stride

        self.pool_w = size
        self.pool_h = size
        self.pad = pad

    def __call__(self, inputs, *args, **kwargs):

        # time1 = time.time()

        feature = inputs

        # pool_out, pool_out_max_location = self._forward_cpu(feature)
        pool_out = self._forward_cpu1(feature)

        # print('pooling:{}'.format(time.time() - time1))

        return pool_out

    def backprop(self, pool_out_delta):

        # 目前默认cpu
        # delta = self._backprop_cpu(pool_out_delta, pool_out_max_location)
        delta = self._backprop_cpu1(pool_out_delta)

        return delta

    def _forward_cpu1(self, x):

        x = np.expand_dims(x, axis=0)
        x = x.transpose(0, 3, 1, 2)

        N, C, H, W = x.shape

        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        # spe(self.pool_h, self.pool_w, out_h, out_w)

        # 展开
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 最大值
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)

        # 转换
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        # 还原shape
        out = out.transpose(0, 2, 3, 1)[0]

        return out

    def _backprop_cpu1(self, dout):

        dout = np.expand_dims(dout, axis=0)
        dout = dout.transpose(0, 3, 1, 2)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        # 还原shape
        dx = dx.transpose(0, 2, 3, 1)[0]

        return dx

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
