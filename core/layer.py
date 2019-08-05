#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from taurus.core.operation import OperationNode
from taurus.core.graph import Node


class Layer(Node):

    id = None

    inputs = None
    outputs = None

    weights = []
    biases = []

    def __init__(self):
        super(Layer, self).__init__()

    def __call__(self, inputs, *args, **kwargs):
        raise NotImplementedError

