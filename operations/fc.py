#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import numpy as np
from taurus.core.layer import Layer
from taurus.operations import sigmoid


class FC(Layer):

    def __init__(self, units, initializer='normal'):

        super(FC, self).__init__()
        self.units = units
        self.in_size = None
        self.out_size = None
        self.weights = None
        self.biases = None
        self.initializer = initializer

    def __call__(self, inputs, *args, **kwargs):

        x = np.asarray(inputs)

        self.in_size = x.size
        self.out_size = self.units

        # print(x.ndim, x.shape, x.size)

        self.weights = np.zeros(shape=(self.out_size, self.in_size))
        self.biases = np.zeros(shape=(self.out_size, 1))

        # 初始化权重
        self._init_weights()

        x = sigmoid(np.dot(self.weights, x) + self.biases)

        return x

    def _init_weights(self):

        if self.initializer == 'normal':
            self.weights = np.random.randn(self.weights.shape)
            self.biases = np.random.randn(self.biases.shape)

