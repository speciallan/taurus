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
            group = self.file.create_group(name=k)
            for k1,v1 in v.items():
                group.create_dataset(name=k1, data=v1)

        return self.file

    def save_weights(self):
        pass


class Loader(object):

    file = None

    def __init__(self, filepath):
        self.file = h5py.File(filepath, 'r')

    def load(self):

        data = {}

        def get_all_data(t):
            ins = self.file[t]
            if isinstance(ins, h5py.Dataset):
                data[t] = ins[:]

        self.file.visit(get_all_data)

        return data

    def load_weights(self):
        pass
