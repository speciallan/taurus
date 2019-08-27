#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
sys.path.append('..')
import argparse

import numpy as np
from taurus.datasets.mnist import load_data
from taurus.models.mlp import MLP, NewMLP
from taurus.models.cnn import CNN, NewCNN
from taurus.models.model import Model
from taurus.optimizers import SGD
from taurus.operations import FC
from taurus.preprocessing.generators import ImageGenerator
from taurus.config import current_config as config
from sklearn.model_selection import train_test_split
from taurus.utils.spe import spe
from taurus.operations.convolution import padding


def new_mlp(args):

    data_path = config.data_path + '/MNIST/'
    model_path = './model_weights.h5'

    (X_train, y_train), _ = load_data(data_path)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # X_train = X_train[:1000]
    # y_train = y_train[:1000]
    # X_valid = X_valid[:1000]
    # y_valid = y_valid[:1000]

    model = NewMLP()
    optimizer = SGD(learning_rate=5)
    model.set_optimizer(optimizer)

    history = model.train(x=X_train,
                          y=y_train,
                          batch_size=30,
                          epochs=args.epochs,
                          x_valid=X_valid,
                          y_valid=y_valid)

    model.save_weights(model_path)

def mlp(args):

    data_path = config.data_path + '/MNIST/'
    model_path = './model.h5'

    (X_train, y_train), _ = load_data(data_path)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # print(X_train.shape, X_valid.shape)

    train_generator = ImageGenerator(X_train, y_train,
                                     batch_size=config.batch_size,
                                     image_size=(512,512))

    optimizer = SGD(learning_rate=5)

    mlp_structure = [784, 30, 10]
    model = MLP(mlp_structure)
    model.set_optimizer(optimizer)

    history = model.train(x=X_train,
                          y=y_train,
                          batch_size=30,
                          epochs=10,
                          x_valid=X_valid,
                          y_valid=y_valid)

    model.save(model_path)


def new_cnn(args):

    data_path = config.data_path + '/MNIST/'
    model_path = './model_weights_cnn.h5'

    (X_train, y_train), _ = load_data(data_path)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    X_train = X_train[:600]
    y_train = y_train[:600]
    X_valid = X_valid[:100]
    y_valid = y_valid[:100]

    # MLP可以用784 做卷积需要28x28
    X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
    X_valid = np.reshape(X_valid, (len(X_valid), 28, 28, 1))

    X_train = padding(X_train, 2)  # 对初始图像进行零填充，保证与LeNet输入结构一致60000*32*32*1
    X_valid = padding(X_valid, 2)

    # 3e-5
    optimizer = SGD(learning_rate=3e-5)
    model = NewCNN()
    model.set_optimizer(optimizer)

    history = model.train(x=X_train,
                          y=y_train,
                          batch_size=100,
                          epochs=10,
                          x_valid=X_valid,
                          y_valid=y_valid)

    model.save_weights(model_path)


def cnn(args):

    data_path = config.data_path + '/MNIST/'
    model_path = './model_weights_cnn.h5'

    (X_train, y_train), _ = load_data(data_path)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    X_train = X_train[:1000]
    y_train = y_train[:1000]
    X_valid = X_valid[:1000]
    y_valid = y_valid[:1000]

    # MLP可以用784 做卷积需要28x28
    X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
    X_valid = np.reshape(X_valid, (len(X_valid), 28, 28, 1))

    X_train = padding(X_train, 2)  # 对初始图像进行零填充，保证与LeNet输入结构一致60000*32*32*1
    X_valid = padding(X_valid, 2)

    optimizer = SGD(learning_rate=0.01)
    model = CNN()
    model.set_optimizer(optimizer)

    history = model.train(x=X_train,
                          y=y_train,
                          batch_size=100,
                          epochs=10,
                          x_valid=X_valid,
                          y_valid=y_valid)

    model.save_weights(model_path)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--new', '-n', default=0, type=int, help='new')
    parser.add_argument('--cnn', '-c', default=0, type=int, help='cnn')
    parser.add_argument('--epochs', '-e', default=10, type=int, help='epochs')
    parser.add_argument('--learning_date', '-lr', default=0.001, type=float, help='learning_rate')
    args = parser.parse_args(sys.argv[1:])

    if args.cnn == 0:
        if args.new == 1:
            new_mlp(args)
        else:
            mlp(args)
    else:
        if args.new == 1:
            new_cnn(args)
        else:
            cnn(args)
