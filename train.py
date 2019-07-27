#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
sys.path.append('..')

from taurus.datasets.mnist import load_data
from taurus.models.mlp import MLP
from taurus.optimizers import SGD
from taurus.preprocessing.generators import ImageGenerator
from taurus.config import current_config as config


if __name__ == '__main__':

    data_path = '/home/speciallan/Documents/python/data/MNIST/'
    weights_path = './model.h5'

    (X_train, y_train), (X_test, y_test) = load_data(data_path)

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
                          epochs=1,
                          x_valid=X_test,
                          y_valid=y_test)

    model.save(weights_path)

