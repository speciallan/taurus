#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

class Initializer(object):
    """初始化器"""

    def __call__(self, shape, dtype=None):
        raise NotImplementedError
