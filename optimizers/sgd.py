#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from taurus import optimizers


class SGD(optimizers.Optimizer):

    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def __call__(self, *args, **kwargs):
        pass

    def optimize(self, x, y, batch_size, epochs):
        pass

