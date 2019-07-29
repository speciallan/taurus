#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
sys.path.append('..')

import numpy as np
from taurus.datasets.mnist import load_data
from taurus.models.mlp import load_model
from taurus.config import current_config as config


def main():

    data_path = config.data_path + '/MNIST/'
    model_path = './model.h5'

    _, (X_test, y_test) = load_data(data_path)

    X_batch = X_test[:]
    y_batch = y_test[:]

    model = load_model(model_path)

    y_pred = model.inference(X_batch)

    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_batch, axis=1)

    true_num = 0
    total = len(y_pred)

    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            true_num += 1

    print('accuracy: {:.2f}%'.format(true_num / total * 100))


if __name__ == '__main__':
    main()