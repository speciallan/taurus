#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from taurus import optimizers


class Model():

    x_batch = None
    y_batch = None

    weights = None
    biases = None

    generator = None
    batch_size = None
    epochs = None

    optimizer = optimizers.SGD()

    def __init__(self, **kwargs):

        if isinstance(self.optimizer, str):
            raise TypeError('类型必须是optimizer')

        elif isinstance(self.optimizer, optimizers.Optimizer):
            pass

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def _forward(self, x):
        raise NotImplementedError

    def _backprop(self, x, y):
        raise NotImplementedError

    def _update(self, x_batch, y_batch):
        raise NotImplementedError

    def _validate(self, x, y):
        raise NotImplementedError

    def train(self, x, y, batch_size, epochs, x_valid, y_valid):

        self.x_batch = x
        self.y_batch = y
        self.batch_size = batch_size
        self.epochs = epochs

    def train_by_generator(self, generator, batch_size, epochs):

        self.generator = generator
        self.batch_size = batch_size
        self.epochs = epochs

    def inference(self, x):
        raise NotImplementedError

    def evaluate(self, x, y):
        raise NotImplementedError

    def save(self, filepath):
        pass

    def save_weigths(self, filepath):
        pass

    def load_weights(self, filepath):
        pass
