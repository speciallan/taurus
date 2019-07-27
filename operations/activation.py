#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import numpy as np

def sigmoid(z):
    a = 1.0 / (1.0 + np.exp(-z))
    return a

def sigmoid_prime(z):
    """sigmoid函数的一阶导数"""
    return sigmoid(z) * (1 - sigmoid(z))
