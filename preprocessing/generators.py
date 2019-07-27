#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan


class Generator(object):

    def __init__(self, x, y):
        pass


class ImageGenerator(Generator):

    def __init__(self, x, y, batch_size, image_size=(512,512)):
        super(ImageGenerator, self).__init__(x, y)

