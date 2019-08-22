#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from taurus import models
from taurus.operations import *
from taurus.operations.convolution import Conv2D, add_bias
from taurus.core.saver import Saver, Loader
from taurus import optimizers
from taurus.utils.spe import spe


class NewCNN(models.BaseModel):

    def __init__(self):
        super(NewCNN, self).__init__()

    def init_weights(self):

        self.batch_size = None
        self.epochs = None
        self.optimizer = optimizers.SGD()

        self.weights = []
        self.biases = []

        self.zs = []
        self.activations = []

        # nodes
        self.input = None

        # layers
        self.layers = []
        self.layer_names = []
        self.layers_avalible = []

        # 定义网络
        self._define()

        # 构建图
        self._build()

    def _build(self):
        """动态构建网络"""
        layer_id = 0
        for name, layer in self.__dict__.items():
            if isinstance(layer, Layer):

                # 赋值id
                layer.id = layer_id
                layer.name = name
                layer_id += 1

                # 层名称
                self.layers.append(layer)
                self.layer_names.append(name)

        # 构建计算图节点，根据前向传播网络结构初始化权重
        self.output = self._forward(self.input)

        # 权重合并
        for name, layer in self.__dict__.items():
            # 不合并输入输出权重
            if isinstance(layer, Layer) and name not in ['input', 'output']:
                # print(layer.id, layer.input_shape, layer.output_shape)
                # print(layer.weights.shape)
                self.layers_avalible.append(layer)
                self.weights.append(layer.weights)
                self.biases.append(layer.biases)

    def _define(self):
        """定义网络结构"""

        self.input = Input(shape=(784,1))
        self.conv1 = Conv2D()
        self.fc1 = FC(units=50, activation='sigmoid', initializer='normal')
        self.fc2 = FC(units=10, activation='sigmoid', initializer='normal')


# ---------------------------------------------------------------------

class CNN(models.BaseModel):

    def __init__(self):
        super(CNN, self).__init__()

        self.filters = [np.random.randn(6, 5, 5, 1)]  # 图像变成 28*28*6 池化后图像变成14*14*6
        self.filters.append(np.random.randn(16, 5, 5, 6))  # 图像变成 10*10*16 池化后变成5*5*16

        self.filters_biases = [np.random.randn(6, 1)]
        self.filters_biases.append(np.random.randn(16, 1))

        # 400 120 84 10
        self.weights = [np.random.randn(120, 400)]
        self.weights.append(np.random.randn(84, 120))
        self.weights.append(np.random.randn(10, 84))

        self.biases = [np.random.randn(120, 1)]
        self.biases.append(np.random.randn(84, 1))
        self.biases.append(np.random.randn(10, 1))

        self.zs = []
        self.activations = []

        self._define()

    def __call__(self, inputs, *args, **kwargs):
        print(inputs)

    def _define(self):

        self.input = Input(shape=(32,32,1))
        # todo
        self.conv1 = Conv2D(filters=self.filters[0], biases=self.filters_biases[0])
        self.pool1 = MaxPooling2D()
        self.conv2 = Conv2D(filters=self.filters[1], biases=self.filters_biases[1])
        self.pool2 = MaxPooling2D()
        self.flatten = Flatten()
        self.fc1 = FC(units=120, activation=None, initializer='normal')
        self.fc2 = FC(units=84, activation=None, initializer='normal')
        self.fc3 = FC(units=10, activation=None, initializer='normal')

    def _forward(self, x):

        # 先前向传播，求出各中间量
        # 第一层卷积 28x28x6 0.003
        conv1 = self.conv1(x)
        # print('conv1:{}'.format(conv1.shape))

        # 0.004
        relu1 = relu(conv1)

        # 14x14x6 很慢 0.01 优化后 0.0001
        pool1 = self.pool1(relu1)
        # print('pool1:{}'.format(pool1.shape))

        # 第二层卷积 10x10x16
        conv2 = self.conv2(pool1)
        # print('conv2:{}'.format(conv2.shape))

        relu2 = relu(conv2)

        # 5x5x16
        pool2 = self.pool2(relu2)
        # print('pool2:{}'.format(pool2.shape))

        # 拉直 400x1
        flatten = self.flatten(pool2)
        # spe(straight_input.shape, self.weights[0].shape)

        # 第一层全连接 120x1
        full_connect1_z = np.dot(self.weights[0], flatten) + self.biases[0]
        full_connect1_a = relu(full_connect1_z)

        # 第二层全连接 84x1
        full_connect2_z = np.dot(self.weights[1], full_connect1_a) + self.biases[1]
        full_connect2_a = relu(full_connect2_z)

        # 第三层全连接（输出） 10x1
        full_connect3_z = np.dot(self.weights[2], full_connect2_a) + self.biases[2]
        full_connect3_a = softmax(full_connect3_z)

        outputs = full_connect3_a

        return outputs

    # todo
    def _backprop(self, x, y):

        '''计算通过单幅图像求得梯度'''
        # time1 = time.time()

        # 先前向传播，求出各中间量
        # 第一层卷积 28x28x6 0.003
        conv1 = self.conv1(x)
        # print('conv1:{}'.format(conv1.shape))

        # 0.004
        relu1 = relu(conv1)

        # 14x14x6 很慢 0.01 优化后 0.0001
        pool1 = self.pool1(relu1)
        # print('pool1:{}'.format(pool1.shape))

        # 第二层卷积 10x10x16
        conv2 = self.conv2(pool1)
        # print('conv2:{}'.format(conv2.shape))

        relu2 = relu(conv2)

        # 5x5x16
        pool2 = self.pool2(relu2)
        # print('pool2:{}'.format(pool2.shape))

        # 拉直 400x1
        flatten = self.flatten(pool2)

        # 第一层全连接 120x1
        fc1_z = self.fc1(flatten)
        fc1_a = relu(fc1_z)

        fc2_z = self.fc2(fc1_a)
        fc2_a = relu(fc2_z)

        fc3_z = self.fc3(fc2_a)
        fc3_a = softmax(fc3_z)

        # print('forward:{}'.format(time.time() - time1))
        # time1 = time.time()

        # 在这里我们使用交叉熵损失，激活函数为softmax，因此delta值就为 a-y，即对正确位置的预测值减1
        cost = delta_fc3 = fc3_a - y

        delta_fc2 = np.dot(self.weights[2].transpose(), delta_fc3) * relu_prime(fc2_z)
        delta_fc1 = np.dot(self.weights[1].transpose(), delta_fc2) * relu_prime(fc1_z)
        delta_flatten = np.dot(self.weights[0].transpose(), delta_fc1)

        delta_pool2 = delta_flatten.reshape(pool2.shape)

        # pooling + relu
        delta_conv2 = relu_prime(conv2) * self.pool2.backprop(delta_pool2)
        # print('delta_conv2:{}'.format(delta_conv2.shape))

        delta_pool1 = self.conv2.backprop(delta_conv2)
        # print('delta_pool1:{}'.format(delta_pool1.shape))
        # delta_pool1 = Conv2D(rot180(self.filters[1]).swapaxes(0,3))(padding(delta_conv2, self.filters[1].shape[1] - 1))

        delta_conv1 = relu_prime(conv1) * self.pool1.backprop(delta_pool1)
        # print('delta_conv1:{}'.format(delta_conv1.shape))


        # 求各参数的导数
        nabla_w2 = np.dot(delta_fc3, fc2_a.transpose())
        nabla_b2 = delta_fc3
        nabla_w1 = np.dot(delta_fc2, fc1_a.transpose())
        nabla_b1 = delta_fc2
        nabla_w0 = np.dot(delta_fc1, flatten.transpose())
        nabla_b0 = delta_fc1

        # time2 = time.time()
        # 计算filter误差 占了反向传播一半时间  x->conv1 pool1->conv2
        nabla_filters1 = conv_cal_w(delta_conv2, pool1)
        nabla_filters0 = conv_cal_w(delta_conv1, x)
        nabla_filters_biases1 = conv_cal_b(delta_conv2)
        nabla_filters_biases0 = conv_cal_b(delta_conv1)
        # print(nabla_filters0.shape, nabla_filters1.shape)

        # print('test:{}'.format(time.time() - time2))

        # 合并conv和fc的权重
        nabla_w = [nabla_w0, nabla_w1, nabla_w2]
        nabla_b = [nabla_b0, nabla_b1, nabla_b2]
        nabla_f = [nabla_filters0, nabla_filters1]
        nabla_fb = [nabla_filters_biases0, nabla_filters_biases1]

        # print('backprop:{}'.format(time.time() - time1))

        return nabla_w, nabla_b, nabla_f, nabla_fb, cost

    def _update(self, x_batch, y_batch):

        '''通过一个batch的数据对神经网络参数进行更新
        需要先求这个batch中每张图片的误差反向传播求得的权重梯度以及偏置梯度'''

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        nabla_f = [np.zeros(f.shape) for f in self.filters]
        nabla_fb = [np.zeros(fb.shape) for fb in self.filters_biases]

        cost_all = 0
        i = 0
        for x, y in zip(x_batch, y_batch):

            # time1 = time.time()
            # print('--------', i)
            delta_nabla_w, delta_nabla_b, delta_nabla_f, delta_nabla_fb, cost = self._backprop(x, y)
            # print('{} backprop_total:{}'.format(i, time.time() - time1))
            i += 1

            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_f = [nf + dnf for nf, dnf in zip(nabla_f, delta_nabla_f)]
            nabla_fb = [nfb + dnfb for nfb, dnfb in zip(nabla_fb, delta_nabla_fb)]
            cost_all += sum(abs(cost))[0]

        self.weights = [w - (self.optimizer.learning_rate / self.batch_size) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (self.optimizer.learning_rate / self.batch_size) * nb for b, nb in zip(self.biases, nabla_b)]
        self.filters = [f - (self.optimizer.learning_rate / self.batch_size) * nf for f, nf in zip(self.filters, nabla_f)]
        self.filters_biases = [fb - (self.optimizer.learning_rate / self.batch_size) * nfb for fb, nfb in zip(self.filters_biases, nabla_fb)]

        # 更新全局权重到每一层 conv fc
        self.conv1.W = self.filters[0].transpose(0, 3, 1, 2)
        self.conv1.b = self.filters_biases[0].transpose(1, 0)
        self.conv2.W = self.filters[1].transpose(0, 3, 1, 2)
        self.conv2.b = self.filters_biases[1].transpose(1, 0)

        # spe(self.conv1.W.shape, self.filters[0].shape, self.conv1.b.shape, self.filters_biases[0].shape)

        return cost_all

    def train(self, x, y, batch_size, epochs, x_valid=[], y_valid=[]):

        self.x_batch = x
        self.y_batch = y
        self.batch_size = batch_size
        self.epochs = epochs

        # self.optimizer.optimize(x, y, batch_size, epochs)

        accuracy_history = []
        loss_history = []
        cost = 0

        batch_num = 0
        for j in range(epochs):

            x_batches = [x[k:k + self.batch_size] for k in range(0, len(x), self.batch_size)]
            y_batches = [y[k:k + self.batch_size] for k in range(0, len(y), self.batch_size)]

            for x_batch, y_batch in zip(x_batches, y_batches):
                batch_num += 1
                if batch_num * self.batch_size > len(x):
                    batch_num = 1

                # 12s
                # time1 = time.time()
                cost = self._update(x_batch, y_batch)
                # print('update:{}'.format(time.time() - time1))

                # if batch_num % 100 == 0:
                # print("after {0} training batch: accuracy is {1}/{2}".format(batch_num, self.evaluate(train_image[0:1000], train_label[0:1000]), len(train_image[0:1000])))

                print("\rEpoch{0}:{1}/{2}".format(j + 1, batch_num * self.batch_size, len(x)), end=' ')

            print("After epoch{0}: train_acc is {1}/{2}, val_acc is {3}/{4}, lost is {5:.4f}".format(j + 1, self._evaluate(x, y), len(y_valid), self._evaluate(x_valid, y_valid), len(y_valid), cost))

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

    def save_weights(self, filepath):

        weights = {}

        for k,v in enumerate(self.weights):
            weights['w_' + str(k)] = v

        for k,v in enumerate(self.biases):
            weights['b_' + str(k)] = v

        for k,v in enumerate(self.filters):
            weights['fw_' + str(k)] = v

        for k,v in enumerate(self.filters_biases):
            weights['fb_' + str(k)] = v

        map = {'structure': {'structure': []},
               'weights': weights}

        saver = Saver(filepath)
        saver.save(map)
        print('save weights to {}'.format(filepath))

    def load_weights(self, filepath):

        loader = Loader(filepath)
        data = loader.load()

        # 解析数据
        structure = data['structure/structure']
        weigths, biases, filters, filters_biases = [], [], [], []

        for k,v in data.items():
            if k.startswith('weights/w_'):
                weigths.append(v)
            if k.startswith('weights/b_'):
                biases.append(v)
            if k.startswith('weights/fw_'):
                filters.append(v)
            if k.startswith('weights/fb_'):
                filters_biases.append(v)

        # 加载权重
        self._load_weights(weigths, biases, filters, filters_biases)

    def _load_weights(self, weights, biases, filters, filters_biases):

        self.weights = weights
        self.biases = biases
        self.filters = filters
        self.filters_biases = filters_biases

