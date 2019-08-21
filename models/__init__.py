#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import numpy as np
from taurus import optimizers
from taurus.operations import sigmoid, sigmoid_prime, FC, Input
from taurus.core.layer import Layer
from taurus.core.saver import Saver, Loader
from taurus.utils.spe import spe


class BaseModel(object):

    def __init__(self, **kwargs):

        self.x_batch = None
        self.y_batch = None
        self.generator = None
        self.batch_size = None
        self.epochs = None
        self.optimizer = optimizers.SGD()

        self.weights = []
        self.biases = []
        self.filters = []
        self.filters_biases = []

        # nodes
        self.input = None

        # layers
        self.layers = []
        self.layer_names = []
        self.layers_avalible = []

        if isinstance(self.optimizer, str):
            raise TypeError('类型必须是optimizer')

        elif isinstance(self.optimizer, optimizers.Optimizer):
            pass

    def __call__(self, *args, **kwargs):
        pass

    def _define(self):
        """定义网络结构"""
        raise NotImplementedError

    def _build(self):
        """动态构建网络"""
        layer_id = 0
        for name, layer in self.__dict__.items():
            if isinstance(layer, Layer):

                # 赋值id
                layer.id = layer_id
                layer.name = name
                layer_id += 1

                # 层名称
                self.layers.append(layer)
                self.layer_names.append(name)

        # 构建计算图节点，根据前向传播网络结构初始化权重
        self.output = self._forward(self.input)

        # 权重合并
        for name, layer in self.__dict__.items():
            # 不合并输入输出权重
            if isinstance(layer, Layer) and name not in ['input', 'output']:
                # print(layer.id, layer.input_shape, layer.output_shape)
                # print(layer.weights.shape)
                self.layers_avalible.append(layer)
                self.weights.append(layer.weights)
                self.biases.append(layer.biases)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def _forward(self, x):
        """前向传播"""
        raise NotImplementedError

    def _backprop(self, x, y):

        '''计算通过单幅图像求得的每层权重和偏置的梯度'''
        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 前向传播，计算各层的激活前的输出值以及激活之后的输出值，为下一步反向传播计算作准备
        activations = [x]
        zs = []

        # todo
        for i, layer in enumerate(self.layers_avalible):
            w, b = self.weights[i], self.biases[i]
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            activation = layer.activation_func(z)
            activations.append(activation)

        # 先求最后一层的delta误差以及b和W的导数
        cost = activations[-1] - y
        last_layer = self.layers_avalible[-1]
        delta = cost * last_layer.activation_prime_func(zs[-1])
        delta_nabla_b[-1] = delta
        delta_nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 将delta误差反向传播以及各层b和W的导数，一直计算到第二层
        for l in range(2, len(self.layers_avalible) + 1):
            prime_func = self.layers_avalible[-l+1].activation_prime_func
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * prime_func(zs[-l])
            delta_nabla_b[-l] = delta
            delta_nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return delta_nabla_b, delta_nabla_w, cost

    def _update(self, x_batch, y_batch):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # spe(len(nabla_w), nabla_w[0].shape, nabla_w[1].shape)

        # 把所有的delta误差求和
        cost_all = 0
        for x, y in zip(x_batch, y_batch):

            delta_nabla_b, delta_nabla_w, cost = self._backprop(x, y)

            # 误差求和
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            cost_all += sum(abs(cost))[0]

            # print(np.sum(delta_nabla_w[0][:][:]))
            # print(np.sum(delta_nabla_b[0][:][:]))
            # print('--------------------')

        self.weights, self.biases = self.optimizer.optimize(self.weights, self.biases, nabla_w, nabla_b, len(x_batch))

        # 更新权重到每一个节点
        for i, layer in enumerate(self.layers_avalible):
            layer.weights = self.weights[i]
            layer.biases = self.biases[i]

        return cost_all

    def train(self, x, y, batch_size, epochs, x_valid=[], y_valid=[]):

        self.x_batch = x
        self.y_batch = y
        self.batch_size = batch_size
        self.epochs = epochs

        accuracy_history = []
        loss_history = []
        cost = 0

        for i in range(epochs):

            x_batches = [x[k:k + batch_size] for k in range(0, len(x), batch_size)]
            y_batches = [y[k:k + batch_size] for k in range(0, len(y), batch_size)]
            # spe(len(x_batches), x_batches[0].shape)

            for x_batch, y_batch in zip(x_batches, y_batches):
                cost = self._update(x_batch, y_batch)

            (x_eval, y_eval) = (x, y) if len(x_valid) == 0 or len(y_valid) == 0 else (x_valid, y_valid)

            train_result = self.evaluate(x, y)
            valid_result = self.evaluate(x_eval, y_eval)
            accuracy_history.append(valid_result)

            print("Epoch {0}: train acc is {1}/{2}={3:.2f}%, valid acc is {4}/{5}={6:.2f}%, loss is {7:.4f}".format(i + 1, train_result, len(x), train_result/len(x) * 100, valid_result, len(x_valid), valid_result / len(x_valid) * 100, cost))

        return (accuracy_history, loss_history)

    def inference(self, x_batch):

        y_batch = np.array([])

        for i in range(len(x_batch)):
            y = self._forward(x_batch[i])
            if i == 0:
                y_batch = np.array([np.zeros(shape=(y.shape))])
            y_batch = np.append(y_batch, [y], axis=0)
        y_batch = np.delete(y_batch, [0], axis=0)

        return y_batch

    def evaluate(self, x, y):

        result = 0
        for img, label in zip(x, y):
            predict_label = self._forward(img)
            if np.argmax(predict_label) == np.argmax(label):
                result += 1

        return result

    def _validate(self, x, y):
        raise NotImplementedError

    def train_by_generator(self, generator, batch_size, epochs):

        self.generator = generator
        self.batch_size = batch_size
        self.epochs = epochs

    def save(self, filepath):
        raise NotImplementedError

    def save_weigths(self, filepath):
        pass

    def load_weights(self, filepath):
        pass

    def _load_weights(self, weights, biases):
        self.weights = weights
        self.biases = biases
