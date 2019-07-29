#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys

import numpy as np
from taurus import operations


class Conv(operations.Operation):

    def __init__(self):
        super(Conv, self).__init__()


class Conv2D(Conv):

    def __init__(self, conv_filter, stride=1, padding='valid', biases=None):
        super(Conv2D, self).__init__()

        # self.filters = filters
        # self.kernel_size = kernel_size
        self.conv_filter = conv_filter
        self.stride = stride
        self.padding = padding
        self.biases = biases

    def __call__(self, img, *args, **kwargs):

        if len(img.shape) != 3 or len(self.conv_filter.shape) != 4:
            print("卷积运算所输入的维度不符合要求")
            sys.exit()

        if img.shape[-1] != self.conv_filter.shape[-1]:
            print("卷积输入图片与卷积核的通道数不一致")
            sys.exit()

        img_h, img_w, img_ch = img.shape
        filter_num, filter_h, filter_w, img_ch = self.conv_filter.shape
        feature_h = img_h - filter_h + 1
        feature_w = img_w - filter_w + 1

        # 初始化输出的特征图片，由于没有使用零填充，图片尺寸会减小
        img_out = np.zeros((feature_h, feature_w, filter_num))
        img_matrix = np.zeros((feature_h * feature_w, filter_h * filter_w * img_ch))
        filter_matrix = np.zeros((filter_h * filter_w * img_ch, filter_num))

        # 将输入图片张量转换成矩阵形式
        for i in range(feature_h * feature_w):
            for j in range(img_ch):
                img_matrix[i, j * filter_h * filter_w:(j + 1) * filter_h * filter_w] = \
                    img[np.uint16(i / feature_w):np.uint16(i / feature_w + filter_h),
                    np.uint16(i % feature_w):np.uint16(i % feature_w + filter_w), j].reshape(filter_h * filter_w)

        # 将卷积核张量转换成矩阵形式
        for i in range(filter_num):
            filter_matrix[:, i] = self.conv_filter[i, :].reshape(filter_w * filter_h * img_ch)

        feature_matrix = np.dot(img_matrix, filter_matrix)

        # 将以矩阵形式存储的卷积结果再转换为张量形式
        for i in range(filter_num):
            img_out[:, :, i] = feature_matrix[:, i].reshape(feature_h, feature_w)

        # 添加偏置项
        if self.biases != None:
            img_out = add_bias(img_out, self.biases)

        return img_out

def padding(image, zero_num):
    if len(image.shape) == 4:
        image_padding = np.zeros(
            (image.shape[0], image.shape[1] + 2 * zero_num, image.shape[2] + 2 * zero_num, image.shape[3]))
        image_padding[:, zero_num:image.shape[1] + zero_num, zero_num:image.shape[2] + zero_num, :] = image
    elif len(image.shape) == 3:
        image_padding = np.zeros((image.shape[0] + 2 * zero_num, image.shape[1] + 2 * zero_num, image.shape[2]))
        image_padding[zero_num:image.shape[0] + zero_num, zero_num:image.shape[1] + zero_num, :] = image
    else:
        print("维度错误")
        sys.exit()
    return image_padding

def rot180(conv_filters):
    rot180_filters = np.zeros((conv_filters.shape))
    for filter_num in range(conv_filters.shape[0]):
        for img_ch in range(conv_filters.shape[-1]):
            rot180_filters[filter_num, :, :, img_ch] = np.flipud(np.fliplr(conv_filters[filter_num, :, :, img_ch]))
    return rot180_filters

def add_bias(conv, bias):
    if conv.shape[-1] != bias.shape[0]:
        print("给卷积添加偏置维度出错")
    else:
        for i in range(bias.shape[0]):
            conv[:, :, i] += bias[i, 0]
    return conv

def conv_cal_w(out_img_delta, in_img):
    # 由卷积前的图片以及卷积后的delta计算卷积核的梯度
    nabla_conv = np.zeros([out_img_delta.shape[-1], in_img.shape[0] - out_img_delta.shape[0] + 1,
                           in_img.shape[1] - out_img_delta.shape[1] + 1, in_img.shape[-1]])
    for filter_num in range(nabla_conv.shape[0]):
        for ch_num in range(nabla_conv.shape[-1]):
            nabla_conv[filter_num, :, :, ch_num] = conv_(in_img[:, :, ch_num], out_img_delta[:, :, filter_num])
    return nabla_conv


def conv_cal_b(out_img_delta):
    nabla_b = np.zeros((out_img_delta.shape[-1], 1))
    for i in range(out_img_delta.shape[-1]):
        nabla_b[i] = np.sum(out_img_delta[:, :, i])
    return nabla_b

def conv_(img, conv_filter):
    # 对二维图像以及二维卷积核进行卷积，不填充
    img_h, img_w = img.shape
    filter_h, filter_w = conv_filter.shape
    feature_h = img_h - filter_h + 1
    feature_w = img_w - filter_w + 1

    img_matrix = np.zeros((feature_h * feature_w, filter_h * filter_w))
    for i in range(feature_h * feature_w):
        img_matrix[i:] = img[np.uint16(i / feature_w):np.uint16(i / feature_w + filter_h),
                         np.uint16(i % feature_w):np.uint16(i % feature_w + filter_w)].reshape(filter_w * filter_h)
    filter_matrix = conv_filter.reshape(filter_h * filter_w, 1)

    img_out = np.dot(img_matrix, filter_matrix)

    img_out = img_out.reshape(feature_h, feature_w)

    return img_out

