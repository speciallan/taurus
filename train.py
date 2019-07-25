#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os
import sys
sys.path.append('../../..')

from taurus.initializers.initializer import Ones, Initializer


if __name__ == '__main__':

    ones = Ones()
    init = Initializer()
    # init(shape=(1,1))
    print(ones(shape=(2,2)))