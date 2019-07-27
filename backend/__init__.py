#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from __future__ import absolute_import

import os
import sys


BACKEND = 'taurus'
# BACKEND = 'tensorflow'

if 'TAURUS_BACKEND' in os.environ:
    backend = os.environ['TAURUS_BACKEND']
    if backend:
        BACKEND = backend

if BACKEND == 'taurus':
    sys.stdout.write('Using Taurus as backend.\n')
    from .taurus_backend import *

elif BACKEND == 'tensorflow':
    sys.stdout.write('Using Tensorflow as backend.\n')
    from .tensorflow_backend import *

else:
    sys.stdout.write('Using no backend.\n')
    exit()


__version__ = '0.1.0'
