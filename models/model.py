#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from taurus import models

import numpy as np


class Model(models.BaseModel):

    def __init__(self):
        super(Model, self).__init__()

        self.filters = [np.random.randn(6, 5, 5, 1)]  # 图像变成 28*28*6 池化后图像变成14*14*6
        self.filters_biases = [np.random.randn(6, 1)]
        self.filters.append(np.random.randn(16, 5, 5, 6))  # 图像变成 10*10*16 池化后变成5*5*16
        self.filters_biases.append(np.random.randn(16, 1))

        self.weights = [np.random.randn(120, 400)]
        self.weights.append(np.random.randn(84, 120))
        self.weights.append(np.random.randn(10, 84))
        self.biases = [np.random.randn(120, 1)]
        self.biases.append(np.random.randn(84, 1))
        self.biases.append(np.random.randn(10, 1))

        # self.define()

    def __call__(self, inputs, *args, **kwargs):
        print(inputs)

