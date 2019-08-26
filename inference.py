#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
sys.path.append('..')
import argparse

import numpy as np
from taurus.datasets.mnist import load_data
from taurus.models.mlp import load_model
from taurus.config import current_config as config
from taurus.models.mlp import MLP, NewMLP
from taurus.models.cnn import CNN, NewCNN
from sklearn.model_selection import train_test_split
from taurus.utils.spe import spe
from taurus.operations.convolution import padding


def mlp_new():

    data_path = config.data_path + '/MNIST/'
    model_path = './model_weights.h5'

    _, (X_test, y_test) = load_data(data_path)

    model = NewMLP()
    model.load_weights(model_path)

    x_batch = X_test[:20]

    y_pred = model.inference(x_batch)
    result = np.argmax(y_pred, axis=1)
    print(result)


def mlp_old():

    data_path = config.data_path + '/MNIST/'
    model_path = './model.h5'

    _, (X_test, y_test) = load_data(data_path)

    model = load_model(model_path)

    x_batch = X_test[:20]

    y_pred = model.inference(x_batch)
    result = np.argmax(y_pred, axis=1)
    print(result)

def cnn_new():

    data_path = config.data_path + '/MNIST/'
    model_path = './model_weights_cnn.h5'

    _, (X_test, y_test) = load_data(data_path)
    x_batch = X_test[:20]

    # MLP可以用784 做卷积需要28x28
    x_batch = np.reshape(x_batch, (len(x_batch), 28, 28, 1))

    x_batch = padding(x_batch, 2)  # 对初始图像进行零填充，保证与LeNet输入结构一致60000*32*32*1

    model = NewCNN()
    model.load_weights(model_path)

    y_pred = model.inference(x_batch)
    result = np.argmax(y_pred, axis=1)
    print(result)

def cnn_old():

    data_path = config.data_path + '/MNIST/'
    model_path = './model_weights_cnn.h5'

    _, (X_test, y_test) = load_data(data_path)
    x_batch = X_test[:20]

    # MLP可以用784 做卷积需要28x28
    x_batch = np.reshape(x_batch, (len(x_batch), 28, 28, 1))

    x_batch = padding(x_batch, 2)  # 对初始图像进行零填充，保证与LeNet输入结构一致60000*32*32*1

    model = CNN()
    model.load_weights(model_path)

    y_pred = model.inference(x_batch)
    result = np.argmax(y_pred, axis=1)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn', '-c', default=0, type=int, help='cnn')
    parser.add_argument('--new', '-n', default=0, type=int, help='new')
    args = parser.parse_args(sys.argv[1:])

    if args.cnn == 0:
        if args.new == 0:
            mlp_old()
        else:
            mlp_new()
    else:
        if args.new == 0:
            cnn_old()
        else:
            cnn_new()

