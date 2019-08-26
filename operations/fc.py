#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import numpy as np
from taurus.core.layer import Layer
from taurus import operations
from taurus.operations import sigmoid, sigmoid_prime, relu, relu_prime
from taurus.utils.spe import spe


class FC(operations.Operation):

    def __init__(self, units, activation=None, initializer='normal'):

        super(FC, self).__init__()

        self.type = self.FC
        self.units = units
        self.in_size = None
        self.out_size = None
        self.weights = None
        self.biases = None
        self.activation = activation
        self.initializer = initializer

        # 记录前向反向传播的输入
        self.input = None
        self.delta = None

        self.activation_func = None
        self.activation_prime_func = None

        self.has_inited = False


    def __call__(self, inputs, *args, **kwargs):

        if isinstance(inputs, Layer):
            x = super(FC, self).__call__(inputs)
        else:
            x = inputs

        x = np.array(x)

        # 初始化权重
        if not self.has_inited:

            self.in_size = x.size
            self.out_size = self.units

            # print(x.ndim, x.shape, x.size)

            self.weights = np.zeros(shape=(self.out_size, self.in_size))
            self.biases = np.zeros(shape=(self.out_size, 1))

            # 初始化
            self._init_weights()
            self.has_inited = True

        # 记录输入
        self.input = x

        # 激活函数
        self.outputs = self._forward_cpu(x)

        if self.activation is not None:

            if self.activation == 'sigmoid':
                self.activation_func = sigmoid
                self.activation_prime_func = sigmoid_prime

            elif self.activation == 'relu':
                self.activation_func = relu
                self.activation_prime_func = relu_prime

            self.outputs = self.activation_func(self.outputs)

        self.output_shape = self.outputs.shape

        # print(self.id, self.name, 'in_nodes:', self.in_bounding_nodes, 'out_nodes:', self.out_bounding_nodes)
        # print('input_shape:{}, output_shape:{}'.format(self.input_shape, self.output_shape))
        # print(inputs.id, inputs.name, self.id, self.name)
        # print(inputs.in_bounding_nodes, inputs.out_bounding_nodes, self.in_bounding_nodes, self.out_bounding_nodes)

        # 区分是构造阶段，还是预测阶段
        if isinstance(inputs, Layer):
            return self
        else:
            return self.outputs

    def backprop(self, delta):

        # 记录梯度 反向传播输入
        self.delta = delta
        delta = self._backprop_cpu(delta)
        return delta

    def cul_prime(self):
        nabla_w = np.dot(self.delta, self.input.transpose())
        nabla_b = self.delta
        return nabla_w, nabla_b

    def _forward_cpu(self, x):
        x = np.dot(self.weights, x) + self.biases
        return x

    def _backprop_cpu(self, delta):
        delta = np.dot(self.weights.transpose(), delta)
        return delta

    def _init_weights(self):

        if self.initializer == 'normal':
            self.weights = np.random.randn(self.weights.shape[0], self.weights.shape[1])
            self.biases = np.random.randn(self.biases.shape[0], self.biases.shape[1])

