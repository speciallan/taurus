#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import numpy as np
from taurus.core.operation import OperationNode
from taurus.core.graph import Node


class Layer(Node):

    INPUT = 'input'
    FC = 'fc'
    CONV = 'conv'
    POOLING = 'pooling'

    def __init__(self):
        super(Layer, self).__init__()

        self.id = None
        self.name = None
        self.type = None

        self.inputs = None
        self.outputs = None

        self.input_shape = None
        self.output_shape = None

        self.in_bounding_nodes = []
        self.out_bounding_nodes = []

        self.weights = np.array([])
        self.biases = np.array([])

    def __call__(self, inputs, *args, **kwargs):
        """处理layer的节点信息"""

        layer = inputs
        self.inputs = layer.outputs
        self.input_shape = layer.output_shape

        # 节点信息
        layer.out_bounding_nodes.append(self.id)
        self.in_bounding_nodes.append(layer.id)

        return layer.outputs

