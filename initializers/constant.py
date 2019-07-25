#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from taurus import backend
from taurus import initializers


class Zeros(initializers.Initializer):
    """初始化成零张量"""

    def __call__(self, shape, dtype=None):
        return backend.constant(0, shape=shape, dtype=dtype)


class Ones(initializers.Initializer):
    """初始化成一张量"""

    def __call__(self, shape, dtype=None):
        return backend.constant(1, shape=shape, dtype=dtype)


class Constant(initializers.Initializer):
    """初始化成常量"""

    def __init__(self, value=0):
        self.value = value

    def __call__(self, shape, dtype=None):
        return backend.constant(self.value, shape=shape, dtype=dtype)
