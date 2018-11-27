#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

# 金牛座 Taurus
# To create a flexible and easy framework for myself in deep learning.

# 现象
# 1、数据量小的时候，loss能训练成负数
# 2、

# 需要做
# 1、可视化
# 2、多数据训练
# 3、

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.python.keras import layers, models
from spe import *

NN_ARCHITECTURE = [
    {'input_dim': 2, 'output_dim': 25, 'activation': 'relu'}, # 25*2 2*1 = 25*1
    {'input_dim': 25, 'output_dim': 50, 'activation': 'relu'}, # 50*25 25*1 = 50*1
    {'input_dim': 50, 'output_dim': 50, 'activation': 'relu'}, # 50*50 50*1 = 50*1
    {'input_dim': 50, 'output_dim': 25, 'activation': 'relu'}, # 25*50 50*1 = 25*1
    {'input_dim': 25, 'output_dim': 1, 'activation': 'sigmoid'}, # 1*25 25*1 = 1*1
]

# 初始化网络参数
def init_layers(nn_architecture):

    np.random.seed(len(nn_architecture) * 4) # 大于2的数
    params = {}

    for i, layer in enumerate(nn_architecture):

        layer_input_dim = layer['input_dim']
        layer_output_dim = layer['output_dim']

        params['W_' + str(i)] = np.random.randn(layer_output_dim, layer_input_dim) * 0.1
        params['b_' + str(i)] = np.random.randn(layer_output_dim, 1) * 0.1

    return params


# sigmoid
def sigmoid(X):
    return 1/(1 + np.exp(-X))

# sigmoid求导
def sigmoid_backward(dA, X):
    sig = sigmoid(X)
    return dA * sig * (1 - sig)

# relu
def relu(X):
    return np.maximum(0, X)

# relu求导
def relu_backward(dA, X):
    dA = np.array(dA, copy=True)
    dA[X <= 0] = 0
    return dA

# 损失函数 交叉熵
def get_cost_value(Y_hat, Y):

    m = Y_hat.shape[1]
    # print(Y_hat, 1-Y_hat)
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

# an auxiliary function that converts probability into class
def convert_prob_into_class(probs):

    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0

    return probs_

# 精度函数
def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()

# 单层前向传播
def forward_single_layer(X, W, b, activation='relu'):

    # print('layer：', X.shape, W.shape, b.shape)
    Y = np.dot(W, X) + b

    if activation == 'relu':
        activation_func = relu
    elif activation == 'sigmoid':
        activation_func = sigmoid
    else:
        raise Exception('The activation function' + activation + ' not exists.')

    return activation_func(Y), Y

# 前向传播
def forward(X, params, nn_architecture):

    memory = {}
    X_next = X

    for i, layer in enumerate(nn_architecture):

        X_current = X_next

        activation_func = layer['activation']

        W_current = params['W_' + str(i)]
        b_current = params['b_' + str(i)]

        X_next, Y_current = forward_single_layer(X_current, W_current, b_current, activation_func)

        memory['X_' + str(i)] = X_current
        memory['Y_' + str(i)] = Y_current

    # 返回往下传播的X和参数字典
    return X_next, memory

# 单层反向传播
def backward_single_layer(dX_current, W_current, b_current, Y_current, X_previous, activation='relu'):

    m = X_previous.shape[1]

    if activation == 'relu':
        backward_activation_fun = relu_backward
    elif activation == 'sigmoid':
        backward_activation_fun = sigmoid_backward
    else:
        raise Exception('The backward activation function' + activation + ' not exists.')

    dY_current = backward_activation_fun(dX_current, Y_current)
    dW_current = np.dot(dY_current, X_previous.T) / m
    db_current = np.sum(dY_current, axis=1, keepdims=True) / m
    dX_previous = np.dot(W_current.T, dX_current)

    return dX_previous, dW_current, db_current

# 反向传播
def backward(Y_hat, Y, memory, params_values, nn_architecture):

    gradients = {}
    # m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)

    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):

        layer_idx_curr = layer_idx_prev

        activ_function_curr = layer["activation"]

        dA_curr = dA_prev

        A_prev = memory['X_' + str(layer_idx_prev)]
        Z_curr = memory['Y_' + str(layer_idx_curr)]
        W_curr = params_values['W_' + str(layer_idx_curr)]
        b_curr = params_values['b_' + str(layer_idx_curr)]

        dA_prev, dW_curr, db_curr = backward_single_layer(dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

        gradients['dW_' + str(layer_idx_curr)] = dW_curr
        gradients['db_' + str(layer_idx_curr)] = db_curr

    return gradients

# 更新参数
def update(params_values, grads_values, nn_architecture, learning_rate):

    # iteration over network layers
    for i, layer in enumerate(nn_architecture):

        params_values['W_' + str(i)] -= learning_rate * grads_values['dW_' + str(i)]
        params_values['b_' + str(i)] -= learning_rate * grads_values['db_' + str(i)]

    return params_values

# 训练网络
def train(X, Y, nn_architecture, epochs, learning_rate):

    params_values = init_layers(nn_architecture)
    cost_history = []
    accuracy_history = []

    for i in range(epochs):

        Y_hat, cashe = forward(X, params_values, nn_architecture)

        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)

        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)

        grads_values = backward(Y_hat, Y, cashe, params_values, nn_architecture)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)

        print('Epoch ' + str(i+1) + '/' + str(epochs))
        print('1/1', end=' ')
        print('' + 'loss:', cost, 'acc:', accuracy)

    return params_values, cost_history, accuracy_history

def run_my_network(X_train, y_train, X_test, y_test):

    print('\n----------------------network----------------------')

    params, cost_history, accuracy_history = train(X_train, y_train, NN_ARCHITECTURE, epochs=10, learning_rate=0.01)
    # print('cost:', cost_history, '\n', 'acc:', accuracy_history)

    # 可视化
    # plt.imshow()

def run_keras(X_train, y_train, X_test, y_test):

    print('\n-----------------------keras-----------------------')

    # Building a model
    model = models.Sequential()
    model.add(layers.Dense(25, input_dim=2, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(25, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="sgd", metrics=['accuracy'])

    # Training
    # print(X_train.shape, y_train.shape)

    history = model.fit(X_train, y_train, epochs=10, verbose=1)
    print(model.summary())

    # Y_test_hat = model.predict_classes(X_test)
    # acc_test = accuracy_score(y_test, Y_test_hat)
    # print("Test set accuracy: {:.2f} - Goliath".format(acc_test))

if __name__ == '__main__':

    # number of samples in the data set
    N_SAMPLES = 1000

    # ratio between training and test sets
    TEST_SIZE = 0.1

    # my network
    X = np.random.randn(N_SAMPLES, 2, 1)
    y = np.random.randn(N_SAMPLES, 1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # 先选取第一条
    X_train_0, X_test_0, y_train_0, y_test_0 = X_train[0], X_test[0], y_train[0], y_test[0]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    run_my_network(X_train_0, y_train_0, X_test_0, y_test_0)

    # keras 使用同一个训练集
    # X = np.random.randn(N_SAMPLES, 2)
    # y = np.random.randn(N_SAMPLES, 1)
    X = X[:,:,0]
    y = y[:,:,0]
    # print(X.shape, y.shape)

    # X, y = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, np.array([X_train[0]]).shape)

    run_keras(np.array([X_train[0]]), np.array([y_train[0]]), np.array([X_test[0]]), np.array([y_test[0]]))
    # run_keras(X_train, y_train, X_test, y_test)


