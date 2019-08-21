#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from taurus import models
from taurus.operations import *
from taurus.operations.convolution import Conv2D, add_bias
from taurus.utils.spe import spe


class NewCNN(models.BaseModel):

    def __init__(self):
        super(NewCNN, self).__init__()

    def init_weights(self):

        # conv1
        self.weights.append(np.random.randn(6, 5, 5, 1))
        self.biases.append(np.random.rand(6, 1))

        # conv2
        self.weights.append(np.random.randn(16, 5, 5, 6))
        self.biases.append(np.random.randn(16, 1))

        # fc1
        self.weights.append(np.random.randn(120, 400))
        self.biases.append(np.random.randn(120, 1))

        # fc2
        self.weights.append(np.random.randn(84, 120))
        self.biases.append(np.random.randn(84, 1))

        # fc3
        self.weights.append(np.random.randn(10, 84))
        self.biases.append(np.random.randn(10, 1))

class CNN(models.BaseModel):

    def __init__(self):
        super(CNN, self).__init__()

        self.filters = [np.random.randn(6, 5, 5, 1)]  # 图像变成 28*28*6 池化后图像变成14*14*6
        self.filters_biases = [np.random.randn(6, 1)]
        self.filters.append(np.random.randn(16, 5, 5, 6))  # 图像变成 10*10*16 池化后变成5*5*16
        self.filters_biases.append(np.random.randn(16, 1))

        self.weights = [np.random.randn(120, 400)]
        self.weights.append(np.random.randn(84, 120))
        self.weights.append(np.random.randn(10, 84))
        self.biases = [np.random.randn(120, 1)]
        self.biases.append(np.random.randn(84, 1))
        self.biases.append(np.random.randn(10, 1))

        # self.define()

    def __call__(self, inputs, *args, **kwargs):
        print(inputs)

    def define(self):

        inputs = np.zeros((28,28,1))
        x = MaxPooling2D()(inputs)
        outputs = x
        print(outputs.shape)

        return inputs, outputs


    def _forward(self, x):

        # 第一层卷积
        conv1 = Conv2D(self.filters[0], self.filters_biases[0])(x)

        relu1 = relu(conv1)
        pool1, pool1_max_locate = MaxPooling2D()(relu1)

        # 第二层卷积
        conv2 = Conv2D(self.filters[1], self.filters_biases[1])(pool1)
        relu2 = relu(conv2)
        pool2, pool2_max_locate = MaxPooling2D()(relu2)

        # 拉直
        straight_input = pool2.reshape(pool2.shape[0] * pool2.shape[1] * pool2.shape[2], 1)

        # 第一层全连接
        full_connect1_z = np.dot(self.weights[0], straight_input) + self.biases[0]
        full_connect1_a = relu(full_connect1_z)

        # 第二层全连接
        full_connect2_z = np.dot(self.weights[1], full_connect1_a) + self.biases[1]
        full_connect2_a = relu(full_connect2_z)

        # 第三层全连接（输出）
        full_connect3_z = np.dot(self.weights[2], full_connect2_a) + self.biases[2]
        full_connect3_a = softmax(full_connect3_z)

        outputs = full_connect3_a

        return outputs

    def _backprop(self, x, y):

        '''计算通过单幅图像求得梯度'''

        # 先前向传播，求出各中间量
        # 第一层卷积
        # 第一层卷积
        conv1 = Conv2D(self.filters[0], self.filters_biases[0])(x)

        relu1 = relu(conv1)
        pool1, pool1_max_locate = MaxPooling2D()(relu1)

        # 第二层卷积
        conv2 = Conv2D(self.filters[1], self.filters_biases[1])(pool1)
        relu2 = relu(conv2)
        pool2, pool2_max_locate = MaxPooling2D()(relu2)

        # 拉直
        straight_input = pool2.reshape(pool2.shape[0] * pool2.shape[1] * pool2.shape[2], 1)
        # spe(straight_input.shape, self.weights[0].shape)

        # 第一层全连接
        full_connect1_z = np.dot(self.weights[0], straight_input) + self.biases[0]
        full_connect1_a = relu(full_connect1_z)

        # 第二层全连接
        full_connect2_z = np.dot(self.weights[1], full_connect1_a) + self.biases[1]
        full_connect2_a = relu(full_connect2_z)

        # 第三层全连接（输出）
        full_connect3_z = np.dot(self.weights[2], full_connect2_a) + self.biases[2]
        full_connect3_a = softmax(full_connect3_z)

        # 在这里我们使用交叉熵损失，激活函数为softmax，因此delta值就为 a-y，即对正确位置的预测值减1
        delta_fc3 = full_connect3_a - y
        delta_fc2 = np.dot(self.weights[2].transpose(), delta_fc3) * relu_prime(full_connect2_z)
        delta_fc1 = np.dot(self.weights[1].transpose(), delta_fc2) * relu_prime(full_connect1_z)
        delta_straight_input = np.dot(self.weights[0].transpose(), delta_fc1)  # 这里没有激活函数？
        delta_pool2 = delta_straight_input.reshape(pool2.shape)

        delta_conv2 = MaxPooling2D().backprop(delta_pool2, pool2_max_locate) * relu_prime(conv2)

        delta_pool1 = Conv2D(rot180(self.filters[1]).swapaxes(0,3))(padding(delta_conv2, self.filters[1].shape[1] - 1))

        delta_conv1 = MaxPooling2D().backprop(delta_pool1, pool1_max_locate) * relu_prime(conv1)

        # 求各参数的导数
        nabla_w2 = np.dot(delta_fc3, full_connect2_a.transpose())
        nabla_b2 = delta_fc3
        nabla_w1 = np.dot(delta_fc2, full_connect1_a.transpose())
        nabla_b1 = delta_fc2
        nabla_w0 = np.dot(delta_fc1, straight_input.transpose())
        nabla_b0 = delta_fc1

        nabla_filters1 = conv_cal_w(delta_conv2, pool1)
        nabla_filters0 = conv_cal_w(delta_conv1, x)
        nabla_filters_biases1 = conv_cal_b(delta_conv2)
        nabla_filters_biases0 = conv_cal_b(delta_conv1)

        nabla_w = [nabla_w0, nabla_w1, nabla_w2]
        nabla_b = [nabla_b0, nabla_b1, nabla_b2]
        nabla_f = [nabla_filters0, nabla_filters1]
        nabla_fb = [nabla_filters_biases0, nabla_filters_biases1]

        return nabla_w, nabla_b, nabla_f, nabla_fb

    def _update(self, x_batch, y_batch):

        '''通过一个batch的数据对神经网络参数进行更新
        需要先求这个batch中每张图片的误差反向传播求得的权重梯度以及偏置梯度'''

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        nabla_f = [np.zeros(f.shape) for f in self.filters]
        nabla_fb = [np.zeros(fb.shape) for fb in self.filters_biases]

        for x, y in zip(x_batch, y_batch):

            delta_nabla_w, delta_nabla_b, delta_nabla_f, delta_nabla_fb = self._backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_f = [nf + dnf for nf, dnf in zip(nabla_f, delta_nabla_f)]
            nabla_fb = [nfb + dnfb for nfb, dnfb in zip(nabla_fb, delta_nabla_fb)]

        self.weights = [w - (self.optimizer.learning_rate / self.batch_size) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (self.optimizer.learning_rate / self.batch_size) * nb for b, nb in zip(self.biases, nabla_b)]
        self.filters = [f - (self.optimizer.learning_rate / self.batch_size) * nf for f, nf in zip(self.filters, nabla_f)]
        self.filters_biases = [fb - (self.optimizer.learning_rate / self.batch_size) * nfb for fb, nfb in zip(self.filters_biases, nabla_fb)]

    def _init_weights(self):
        pass

        # self.weights = [np.random.randn(n, m) for m, n in zip(self.sizes[:-1], self.sizes[1:])]
        # self.biases = [np.random.randn(n, 1) for n in self.sizes[1:]]

    def train(self, x, y, batch_size, epochs, x_valid=[], y_valid=[]):

        self.x_batch = x
        self.y_batch = y
        self.batch_size = batch_size
        self.epochs = epochs

        # self.optimizer.optimize(x, y, batch_size, epochs)

        self._init_weights()

        accuracy_history = []
        loss_history = []

        batch_num = 0
        for j in range(epochs):

            x_batches = [x[k:k + self.batch_size] for k in range(0, len(x), self.batch_size)]
            y_batches = [y[k:k + self.batch_size] for k in range(0, len(y), self.batch_size)]

            for x_batch, y_batch in zip(x_batches, y_batches):
                batch_num += 1
                if batch_num * self.batch_size > len(x):
                    batch_num = 1

                self._update(x_batch, y_batch)

                # if batch_num % 100 == 0:
                # print("after {0} training batch: accuracy is {1}/{2}".format(batch_num, self.evaluate(train_image[0:1000], train_label[0:1000]), len(train_image[0:1000])))

                print("\rEpoch{0}:{1}/{2}".format(j + 1, batch_num * self.batch_size, len(x)), end=' ')

            print("After epoch{0}: accuracy is {1}/{2}".format(j + 1, self._evaluate(x_valid, y_valid), len(y_valid)))

        return (accuracy_history, loss_history)

    def inference(self, x_batch):

        y_batch = np.array([])

        for i in range(len(x_batch)):

            y = self._forward(x_batch[i])
            if i == 0:
                y_batch = np.array([np.zeros(shape=(y.shape))])

            y_batch = np.append(y_batch, [y], axis=0)

        y_batch = np.delete(y_batch, [0], axis=0)

        return y_batch

    def _evaluate(self, x, y):

        result = 0
        for img, label in zip(x, y):
            predict_label = self._forward(img)
            if np.argmax(predict_label) == np.argmax(label):
                result += 1

        return result

    def _define(self):
        pass
