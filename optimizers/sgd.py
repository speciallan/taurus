#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from taurus import optimizers


class SGD(optimizers.Optimizer):

    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def __call__(self, *args, **kwargs):
        pass

    def optimize(self, weights, biases, nabla_w, nabla_b, batch_size):

        weights = [w - (self.learning_rate / batch_size) * nw for w, nw in zip(weights, nabla_w)]
        biases = [b - (self.learning_rate / batch_size) * nb for b, nb in zip(biases, nabla_b)]

        return weights, biases

