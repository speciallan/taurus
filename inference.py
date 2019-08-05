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

    model = load_model(model_path)

    x_batch = X_test[:20]

    y_pred = model.inference(x_batch)
    result = np.argmax(y_pred, axis=1)
    print(result)


if __name__ == '__main__':
    main()
