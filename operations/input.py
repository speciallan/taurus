#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import numpy as np
from taurus.core.layer import Layer


class Input(Layer):

    def __init__(self, shape):

        super(Input, self).__init__()

        self.type = self.INPUT
        self.inputs = None
        self.input_shape = None
        self.output_shape = shape
        self.outputs = np.zeros(self.output_shape)

    def __call__(self, inputs, *args, **kwargs):

        super(Input, self).__call__(inputs)
