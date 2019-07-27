#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import h5py


class Saver(object):

    file = None

    def __init__(self, filepath):
        self.file = h5py.File(filepath, 'w')

    def save(self, map):

        for k,v in map.items():
            print(k, v.shape)
            self.file.create_dataset(name=k, data=v)

        return self.file

    def save_weights(self):
        pass


class Loader(object):

    file = None

    def __init__(self, filepath):
        self.file = h5py.File(filepath, 'r')

    def load(self):
        def f(t):
            print(t.name)
        self.file.visit(f)

    def load_weights(self):
        pass
