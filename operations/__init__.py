#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from taurus.core.layer import Layer


class Operation(Layer):

    def __init__(self, name=''):
        super(Operation, self).__init__()


from taurus.operations.input import *
from taurus.operations.activation import *
from taurus.operations.convolution import *
from taurus.operations.fc import *
from taurus.operations.normalization import *
from taurus.operations.pooling import *
from taurus.operations.common import *


