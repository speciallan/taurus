#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import numpy as np

from taurus import models
from taurus.operations.activation import sigmoid, sigmoid_prime
from taurus.core.saver import Saver, Loader


def load_model(filepath):
    loader = Loader(filepath)
    content = loader.load()


class MLP(models.Model):

    def __init__(self, nn_structure, **kwargs):

        super(MLP, self).__init__(**kwargs)

        self.num_layers = len(nn_structure)
        self.sizes = nn_structure
        self.weights = [np.random.randn(n, m) for m, n in zip(nn_structure[:-1], nn_structure[1:])]
        self.biases = [np.random.randn(n, 1) for n in nn_structure[1:]]

    def __call__(self, *args, **kwargs):
        pass

    def forward(self, x):

        value = x
        for i in range(len(self.weights)):
            value = sigmoid(np.dot(self.weights[i], value) + self.biases[i])
        y = value
        return y

    def backprop(self, x, y):

        '''计算通过单幅图像求得的每层权重和偏置的梯度'''
        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 前向传播，计算各层的激活前的输出值以及激活之后的输出值，为下一步反向传播计算作准备
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # 先求最后一层的delta误差以及b和W的导数
        cost = activations[-1] - y
        delta = cost * sigmoid_prime(zs[-1])
        delta_nabla_b[-1] = delta
        delta_nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 将delta误差反向传播以及各层b和W的导数，一直计算到第二层
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(zs[-l])
            delta_nabla_b[-l] = delta
            delta_nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return delta_nabla_b, delta_nabla_w

    def update(self, x_batch, y_batch):

        '''通过一个batch的数据对神经网络参数进行更新
        需要对当前batch中每张图片调用backprop函数将误差反向传播
        求每张图片对应的权重梯度以及偏置梯度，最后进行平均使用梯度下降法更新参数'''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 把所有的delta误差求和
        for x, y in zip(x_batch, y_batch):
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (self.optimizer.learning_rate / self.batch_size) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (self.optimizer.learning_rate / self.batch_size) * nb for b, nb in zip(self.biases, nabla_b)]

    def train(self, x, y, batch_size, epochs, x_valid=[], y_valid=[]):

        self.x_batch = x
        self.y_batch = y
        self.batch_size = batch_size
        self.epochs = epochs

        # self.optimizer.optimize(x, y, batch_size, epochs)

        accuracy_history = []
        loss_history = []

        for i in range(epochs):
            x_batches = [x[k:k + batch_size] for k in range(0, len(x), batch_size)]
            y_batches = [y[k:k + batch_size] for k in range(0, len(y), batch_size)]

            for x_batch, y_batch in zip(x_batches, y_batches):
                self.update(x_batch, y_batch)

            (x_eval, y_eval) = (x, y) if len(x_valid) == 0 or len(y_valid) == 0 else (x_valid, y_valid)

            result = self.evaluate(x_eval, y_eval)
            accuracy_history.append(result)

            print("Epoch {0}: accuracy is {1}/{2}".format(i + 1, result, len(x_eval)))

        return (self.weights, self.biases), (accuracy_history, loss_history)

    def evaluate(self, x, y):

        result = 0

        for img, label in zip(x, y):
            predict_label = self.forward(img)
            if np.argmax(predict_label) == np.argmax(label):
                result += 1

        return result

    def save(self, filepath):

        map = {'structure': self.sizes,
               'weights': np.asarray([self.weights, self.biases])}
        print((map['weights'][0]).shape)
        exit()

        saver = Saver(filepath)
        saver.save(map)

        print('save model to {}'.format(filepath))

    def save_weights(self, filepath):
        print('save weights to {}'.format(filepath))

    def load(self, filepath):

        map = ['structure', 'weights']
        pass


