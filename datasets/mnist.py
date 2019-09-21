#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import numpy as np
import time
from struct import unpack, unpack_from


def load_data(path):
    """
    0.05s by cpu single core
    [todo] multi processing
    """

    # time1 = time.time()

    X_train = read_image(path + 'train-images-idx3-ubyte')
    y_train = read_label(path + 'train-labels-idx1-ubyte')
    X_test = read_image(path + 't10k-images-idx3-ubyte')
    y_test = read_label(path + 't10k-labels-idx1-ubyte')

    X_train = normalize(X_train)
    y_train = one_hot(y_train)
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)

    X_test = normalize(X_test)
    y_test = one_hot(y_test)
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)

    # print(f'load data time:{time.time() - time1}')

    return (X_train, y_train), (X_test, y_test)


def read_image(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784, 1)
    return img


def read_label(path):
    with open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        label = np.fromfile(f, dtype=np.uint8)
    return label


def normalize(image):
    img = image.astype(np.float32) / 255.0
    return img


def one_hot(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab
