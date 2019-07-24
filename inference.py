#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import h5py
import numpy as np
import cv2
import os
from keras.models import load_model

def main():

    filename = 'pkgs.h'

    f = h5py.File(filename, 'r')

    t = list(f.keys())
    w = f['model_weights']
    o = f['optimizer_weights']

    dense_1 = w['dense_1/dense_1']
    kernel_0 = dense_1['kernel:0']
    bias_0 = dense_1['bias:0']

    print(kernel_0.shape, bias_0.shape)

    var_0 = o['Variable:0']
    print(var_0.shape)

    from keras.utils.io_utils import h5dict
    from keras.engine.saving import _deserialize_model
    f = h5dict(filename, mode='r')
    model = _deserialize_model(f)
    model.summary()
    print(f['keras_version'])
    print(f['model_config'])
    exit()
    # print(t, w, o)


    model = load_model(filename)
    # json = model.to_json()
    # print(json)
    # model.summary()

    img_shape = (200,200,3)
    shape = (4,) + img_shape
    img_batch = np.zeros(shape=shape)

    for k,v in enumerate(os.listdir('data')):
        img = cv2.imread('data/' + v)
        img = cv2.resize(img, (200,200))
        img_batch[k] = img

    pred = model.predict(img_batch)
    print(pred)
    # c2 = w['res4f_branch2c']['res4f_branch2c']['kernel:0']
    # print(c2.shape)

if __name__ == '__main__':
    main()

