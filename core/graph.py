#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from taurus.core.operation import OperationNode
from taurus.core.variable import VariableNode


class ComputationalGraph(object):

    def __init__(self, nodes, edges):

        self.nodes = nodes
        self.edges = edges

    def build(self):
        pass


class Node(object):


    def __init__(self):
        self.in_bounding_nodes = []
        self.out_bounding_nodes = []

    def __call__(self, inputs, *args, **kwargs):
        raise NotImplementedError



class Edge():

    def __init__(self):
        pass
