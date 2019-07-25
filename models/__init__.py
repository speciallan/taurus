#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import h5py

from keras.utils.io_utils import h5dict
from keras.engine.saving import _deserialize_model


def load_model(path=None):
    f = h5dict(path, mode='r')
    model = _deserialize_model(f)
    return model