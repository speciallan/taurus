#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
import time

import numpy as np
from taurus import operations
from taurus.operations.common import im2col, col2im
from taurus.core.layer import Layer
from taurus.utils.spe import spe


class Conv(operations.Operation):

    def __init__(self):
        super(Conv, self).__init__()
        self.type = self.CONV


class Conv2D(Conv):

    def __init__(self, filters, kernel_size=3, stride=1, padding='valid', biases=None, pad=0, activation=None, initializer='normal'):
        super(Conv2D, self).__init__()

        self.type = self.CONV

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weights = None
        self.biases = None
        self.pad = pad

        self.activation = activation
        self.initializer = initializer

        # 中间数据（backward时使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

        self.input = None
        self.delta = None

        self.activation_func = None
        self.activation_prime_func = None

        self.has_inited = False

    def __call__(self, inputs, *args, **kwargs):

        if isinstance(inputs, Layer):
            x = super(Conv2D, self).__call__(inputs)
        else:
            x = inputs

        # 初始化权重
        if not self.has_inited:

            n, h, w, c = x.shape
            self.in_channel = c
            self.out_channel = self.filters

            self.weights = np.zeros(shape=(self.out_channel, self.kernel_size, self.kernel_size, self.in_channel))
            self.biases = np.zeros(shape=(self.out_channel, 1))

            # 初始化
            self._init_weights()
            self.has_inited = True

        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)

        self.input = x
        out = self._forward_cpu(x)

        return out

    def _init_weights(self):

        if self.initializer == 'normal':
            self.weights = np.random.randn(self.out_channel, self.kernel_size, self.kernel_size, self.in_channel)
            self.biases = np.random.randn(self.out_channel, 1)

        # 转换shape
        self.weights = self.weights.transpose(0, 3, 1, 2)
        self.biases = self.biases.transpose(1, 0)

    def backprop(self, delta):

        # 保存反向传播的输出
        self.delta = delta

        # 目前默认cpu
        delta = self._backprop_cpu(delta)

        return delta

    def cal_prime(self):

        nabla_w, nabla_b = [], []

        for i in range(self.delta.shape[0]):
            nabla_w.append(conv_cal_w(self.delta[i], self.input[i]))
            nabla_b.append(conv_cal_b(self.delta[i]))

        nabla_w, nabla_b = np.asarray(nabla_w), np.asarray(nabla_b)
        # print(self.delta.shape, self.input.shape, nabla_w.shape)
        return nabla_w, nabla_b

    def _forward_cpu(self, x):

        # print('w', self.weights.shape, 'b', self.biases.shape)

        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)

        x = x.transpose(0, 3, 1, 2)

        # (1,1,32,32) (6,1,5,5)
        # spe(x.shape, self.weights.shape)

        # 卷积核大小
        FN, C, FH, FW = self.weights.shape

        # 数据数据大小
        N, C, H, W = x.shape
        # print(x.shape, self.weights.shape)

        # 计算输出数据大小
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        # 利用im2col转换为行
        # print(x.shape, self.weights.shape)
        col = im2col(x, FH, FW, self.stride, self.pad)

        # 卷积核转换为列，展开为2维数组
        col_W = self.weights.reshape(FN, -1).T

        # 计算正向传播
        # print(col.shape, col_W.shape, self.biases.shape)
        out = np.dot(col, col_W) + self.biases
        # print(out.shape)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # print(out.shape)

        self.x = x
        self.col = col
        self.col_W = col_W

        out = out.transpose(0, 2, 3, 1)

        return out

    def _backprop_cpu(self, dout):

        # (1,16,10,10)
        if dout.ndim == 3:
            dout = np.expand_dims(dout, axis=0)

        dout = dout.transpose(0, 3, 1, 2)

        # 卷积核大小
        FN, C, FH, FW = self.weights.shape
        # dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
        dout = dout.reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        # print(self.dW.shape, self.db.shape)

        dcol = np.dot(dout, self.col_W.T)

        # 逆转换
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        # (1,6,14,14) -> (1,14,14,6)
        dx = dx.transpose(0, 2, 3, 1)

        return dx

    def _forward_cpu_backup(self, img):

        # time1 = time.time()

        if len(img.shape) != 3 or len(self.filters.shape) != 4:
            print("卷积运算所输入的维度不符合要求")
            sys.exit()

        if img.shape[-1] != self.filters.shape[-1]:
            print("卷积输入图片与卷积核的通道数不一致")
            sys.exit()

        img_h, img_w, img_ch = img.shape
        filter_num, filter_h, filter_w, img_ch = self.filters.shape
        feature_h = img_h - filter_h + 1
        feature_w = img_w - filter_w + 1

        # 初始化输出的特征图片，由于没有使用零填充，图片尺寸会减小
        img_out = np.zeros((feature_h, feature_w, filter_num))
        img_matrix = np.zeros((feature_h * feature_w, filter_h * filter_w * img_ch))
        filter_matrix = np.zeros((filter_h * filter_w * img_ch, filter_num))
        # print(' conv_0:{}'.format(time.time() - time1))

        # 将输入图片张量转换成矩阵形式 【最费时间的地方】 0.015
        for i in range(feature_h * feature_w):
            for j in range(img_ch):
                img_matrix[i, j * filter_h * filter_w:(j + 1) * filter_h * filter_w] = \
                    img[np.uint16(i / feature_w):np.uint16(i / feature_w + filter_h),
                    np.uint16(i % feature_w):np.uint16(i % feature_w + filter_w), j].reshape(filter_h * filter_w)

        # 将卷积核张量转换成矩阵形式
        for i in range(filter_num):
            filter_matrix[:, i] = self.filters[i, :].reshape(filter_w * filter_h * img_ch)
        # print(' conv_1:{}'.format(time.time() - time1))

        feature_matrix = np.dot(img_matrix, filter_matrix)
        # print(' conv_2:{}'.format(time.time() - time1))

        # 将以矩阵形式存储的卷积结果再转换为张量形式
        for i in range(filter_num):
            img_out[:, :, i] = feature_matrix[:, i].reshape(feature_h, feature_w)
        # print(' conv_3:{}'.format(time.time() - time1))

        # 添加偏置项
        # if self.biases != None:
        img_out = add_bias(img_out, self.biases)

        # print('conv:{}'.format(time.time() - time1))

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
    nabla_conv = np.zeros([out_img_delta.shape[-1],
                           in_img.shape[0] - out_img_delta.shape[0] + 1,
                           in_img.shape[1] - out_img_delta.shape[1] + 1,
                           in_img.shape[-1]])
    for filter_num in range(nabla_conv.shape[0]):
        for ch_num in range(nabla_conv.shape[-1]):
            nabla_conv[filter_num, :, :, ch_num] = conv_(in_img[:, :, ch_num], out_img_delta[:, :, filter_num])
    return nabla_conv


def conv_cal_b(out_img_delta):
    nabla_b = np.zeros((out_img_delta.shape[-1], 1))
    for i in range(out_img_delta.shape[-1]):
        nabla_b[i] = np.sum(out_img_delta[:, :, i])
    return nabla_b

def conv_(img, filters):
    # 对二维图像以及二维卷积核进行卷积，不填充
    img_h, img_w = img.shape
    filter_h, filter_w = filters.shape
    feature_h = img_h - filter_h + 1
    feature_w = img_w - filter_w + 1

    img_matrix = np.zeros((feature_h * feature_w, filter_h * filter_w))
    for i in range(feature_h * feature_w):
        img_matrix[i:] = img[np.uint16(i / feature_w):np.uint16(i / feature_w + filter_h),
                         np.uint16(i % feature_w):np.uint16(i % feature_w + filter_w)].reshape(filter_w * filter_h)
    filter_matrix = filters.reshape(filter_h * filter_w, 1)

    img_out = np.dot(img_matrix, filter_matrix)

    img_out = img_out.reshape(feature_h, feature_w)

    return img_out

