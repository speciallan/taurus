#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import tensorflow as tf


FLOAT16 = tf.float16
FLOAT32 = tf.float32


# 常量定义
def constant(value, dtype=None, shape=None, name=None):

    if dtype is None:
        dtype = FLOAT32

    return tf.constant(value, dtype=dtype, shape=shape, name=name)
