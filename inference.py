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
from taurus.utils.spe import spe


def new():

    data_path = config.data_path + '/MNIST/'
    model_path = './model_weights.h5'

    _, (X_test, y_test) = load_data(data_path)

    model = NewMLP()
    model.load_weights(model_path)

    x_batch = X_test[:20]

    y_pred = model.inference(x_batch)
    result = np.argmax(y_pred, axis=1)
    print(result)


def old():

    data_path = config.data_path + '/MNIST/'
    model_path = './model.h5'

    _, (X_test, y_test) = load_data(data_path)

    model = load_model(model_path)

    x_batch = X_test[:20]

    y_pred = model.inference(x_batch)
    result = np.argmax(y_pred, axis=1)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--new', '-n', default=0, type=int, help='epochs')
    args = parser.parse_args(sys.argv[1:])

    if args.new == 1:
        new()
    else:
        old()

