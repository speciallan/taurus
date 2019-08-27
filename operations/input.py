#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import numpy as np
from taurus import operations


class Input(operations.Operation):

    def __init__(self, shape):
        """shape 输入只能是3或者4维"""

        super(Input, self).__init__()

        self.type = self.INPUT
        self.inputs = None

        # 如果是三维张量 (32,32,1) -> (1,32,32,1)
        if len(shape) == 3:
            shape = (1,) + shape

        self.input_shape = None
        self.output_shape = shape
        self.outputs = np.zeros(self.output_shape)

    def __call__(self, inputs, *args, **kwargs):

        super(Input, self).__call__(inputs)
