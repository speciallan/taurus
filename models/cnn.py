#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from taurus import models
from taurus.operations import *
from taurus.operations.convolution import Conv2D, add_bias
from taurus.core.saver import Saver, Loader
from taurus import optimizers
from taurus import losses
from taurus.utils.spe import spe
from taurus.CNN import conv, pool, soft_max, pool_delta_error_bp


class NewCNN(models.BaseModel):

    def __init__(self):
        super(NewCNN, self).__init__()

        self.weights = []
        self.biases = []
        self.filters = []
        self.filters_biases = []

        self.convs = []
        self.zs = []
        self.activations = []

        self._define()

        self._build()

    def __call__(self, inputs, *args, **kwargs):
        print(inputs)

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
        self.output, _ = self._forward(self.input)

        # 权重合并
        for name, layer in self.__dict__.items():
            # 不合并输入输出权重
            # if isinstance(layer, Layer) and name not in ['input', 'output']:
            if isinstance(layer, Layer) and layer.type in ['conv', 'fc']:
                self.layers_avalible.append(layer)

                if layer.type == layer.CONV:
                    self.filters.append(layer.weights.transpose(0, 2, 3, 1))
                    self.filters_biases.append(layer.biases.transpose(1, 0))

                if layer.type == layer.FC:
                    self.weights.append(layer.weights)
                    self.biases.append(layer.biases)

    def _define(self):

        self.input = Input(shape=(32,32,1))

        self.conv1 = Conv2D(filters=6, kernel_size=5, initializer='normal')
        self.relu1 = Relu()
        self.pool1 = MaxPooling2D()

        self.conv2 = Conv2D(filters=16, kernel_size=5, initializer='normal')
        self.relu2 = Relu()
        self.pool2 = MaxPooling2D()

        self.flatten = Flatten()

        self.fc1 = FC(units=120, activation=None, initializer='normal')
        self.fc1_relu = Relu()

        self.fc2 = FC(units=84, activation=None, initializer='normal')
        self.fc2_relu = Relu()

        self.fc3 = FC(units=10, activation=None, initializer='normal')

        self.softmax = Softmax()

    def _forward(self, x):

        # 先前向传播，求出各中间量
        # 第一层卷积 28x28x6 0.003
        conv1 = self.conv1(x)

        # 0.004
        relu1 = self.relu1(conv1)

        # 14x14x6 很慢 0.01 优化后 0.0001
        pool1 = self.pool1(relu1)

        # 第二层卷积 10x10x16
        conv2 = self.conv2(pool1)

        relu2 = self.relu2(conv2)

        # 5x5x16
        pool2 = self.pool2(relu2)

        # 拉直 400x1
        flatten = self.flatten(pool2)

        # 第一层全连接 120x1
        fc1_z = self.fc1(flatten)
        fc1_a = self.fc1_relu(fc1_z)

        fc2_z = self.fc2(fc1_a)
        fc2_a = self.fc2_relu(fc2_z)

        fc3_z = self.fc3(fc2_a)
        fc3_a = self.softmax(fc3_z)

        output = fc3_a

        return output, [pool1, flatten, fc1_a, fc2_a]

    def _backprop(self, x, y):

        '''计算通过单幅图像求得梯度'''
        # time1 = time.time()
        output, arr = self._forward(x)

        # print('forward:{}'.format(time.time() - time1))
        # time1 = time.time()

        # 在这里我们使用交叉熵损失，激活函数为softmax，因此delta值就为 a-y，即对正确位置的预测值减
        cost = losses.L1Distance.fn(output, y)

        val = cost

        layers_reversed = self.layers.copy()
        layers_reversed.reverse()

        for i, layer in enumerate(layers_reversed):
            if layer.name not in ['input', 'output']:

                if layer.type == layer.ACTIVATION:
                    val = val * layer.backprop()
                else:
                    val = layer.backprop(val)

        # delta_fc3 = cost * self.softmax.backprop()
        # delta_fc2 = self.fc3.backprop(delta_fc3) * self.fc2_relu.backprop()
        # delta_fc1 = self.fc2.backprop(delta_fc2) * self.fc1_relu.backprop()
        # delta_fla = self.fc1.backprop(delta_fc1)
        #
        # delta_pool2 = self.flatten.backprop(delta_fla)
        #
        # delta_conv2 = self.pool2.backprop(delta_pool2) * self.relu2.backprop()
        # delta_pool1 = self.conv2.backprop(delta_conv2)
        #
        # delta_conv1 = self.pool1.backprop(delta_pool1) * self.relu1.backprop()
        # delta_x     = self.conv1.backprop(delta_conv1)

        # 计算梯度
        nabla_w, nabla_b, nabla_f, nabla_fb = [], [], [], []

        for i, layer in enumerate(self.layers_avalible):

            if layer.type == layer.CONV:
                prime = layer.cul_prime()
                nabla_f.append(prime[0])
                nabla_fb.append(prime[1])

            if layer.type == layer.FC:
                prime = layer.cul_prime()
                nabla_w.append(prime[0])
                nabla_b.append(prime[1])

        # print('backprop:{}'.format(time.time() - time1))

        return nabla_w, nabla_b, nabla_f, nabla_fb, cost

    def _update(self, x_batch, y_batch):

        '''通过一个batch的数据对神经网络参数进行更新
        需要先求这个batch中每张图片的误差反向传播求得的权重梯度以及偏置梯度'''

        # spe(self.filters[0].shape, self.weights[0].shape)

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
            # spe(nabla_w[0].shape, delta_nabla_w[0].shape)
            # spe(nabla_f[0].shape, delta_nabla_f[0].shape)

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
        self._update_weights_to_avaliable_layers()

        # spe(self.conv1.W.shape, self.filters[0].shape, self.conv1.b.shape, self.filters_biases[0].shape)

        return cost_all

    def train(self, x, y, batch_size, epochs, x_valid=[], y_valid=[]):

        starttime =time.time()

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

            print("After epoch{0}: train_acc is {1}/{2}, val_acc is {3}/{4}, lost is {5:.4f}".format(j + 1, self._evaluate(x, y), len(x), self._evaluate(x_valid, y_valid), len(y_valid), cost))

        print('total time:{:2f} m'.format((time.time() - starttime) / 60))

        return (accuracy_history, loss_history)

    def inference(self, x_batch):

        starttime = time.time()

        y_batch = np.array([])

        for i in range(len(x_batch)):

            y, _ = self._forward(x_batch[i])
            if i == 0:
                y_batch = np.array([np.zeros(shape=(y.shape))])

            y_batch = np.append(y_batch, [y], axis=0)

        y_batch = np.delete(y_batch, [0], axis=0)

        print('inference time:{:4f}'.format(time.time() - starttime))

        return y_batch

    def _evaluate(self, x, y):

        result = 0
        for img, label in zip(x, y):
            predict_label, _ = self._forward(img)
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

        # 更新全局权重到每一层 conv fc
        self._update_weights_to_avaliable_layers()

    def _update_weights_to_avaliable_layers(self):

        # 更新全局权重到每一层 conv fc
        conv_id, fc_id = 0, 0
        for i, layer in enumerate(self.layers_avalible):

            if layer.type == layer.CONV:
                layer.weights = self.filters[conv_id].transpose(0, 3, 1, 2)
                layer.biases = self.filters_biases[conv_id].transpose(1, 0)
                conv_id += 1

            if layer.type == layer.FC:
                layer.weights = self.weights[fc_id]
                layer.biases = self.biases[fc_id]
                fc_id += 1


# ---------------------------------------------------------------------

class CNN(models.BaseModel):

    def __init__(self):
        super(CNN, self).__init__()

        self.weights = []
        self.biases = []
        self.filters = []
        self.filters_biases = []

        self.zs = []
        self.activations = []

        self._define()

        self._build()

    def __call__(self, inputs, *args, **kwargs):
        print(inputs)

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
            # if isinstance(layer, Layer) and name not in ['input', 'output']:
            if isinstance(layer, Layer) and layer.type in ['conv', 'fc']:
                self.layers_avalible.append(layer)

                if layer.type == layer.CONV:
                    self.filters.append(layer.weights.transpose(0, 3, 1, 2))
                    self.filters_biases.append(layer.biases.transpose(1, 0))

                if layer.type == layer.FC:
                    self.weights.append(layer.weights)
                    self.biases.append(layer.biases)

    def _define(self):

        self.input = Input(shape=(32,32,1))
        self.conv1 = Conv2D(filters=6, kernel_size=5, initializer='normal')
        self.pool1 = MaxPooling2D()
        self.conv2 = Conv2D(filters=16, kernel_size=5, initializer='normal')
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
        fc1_z = self.fc1(flatten)
        fc1_a = relu(fc1_z)

        # 第二层全连接 84x1
        fc2_z = self.fc2(fc1_a)
        fc2_a = relu(fc2_z)

        # 第三层全连接（输出） 10x1
        fc3_z = self.fc3(fc2_a)
        fc3_a = softmax(fc3_z)

        outputs = fc3_a

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
        # print(pool2)

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
        # print(cost.shape)
        # print(fc3_a, y)
        # cost = delta_fc3 = losses.CrossEntropy.fn(fc3_a, y)

        delta_fc2 = self.fc3.backprop(delta_fc3) * relu_prime(fc2_z)
        delta_fc1 = self.fc2.backprop(delta_fc2) * relu_prime(fc1_z)
        delta_flatten = self.fc1.backprop(delta_fc1)

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
        nabla_filters_biases1 = conv_cal_b(delta_conv2)
        nabla_filters0 = conv_cal_w(delta_conv1, x)
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
            # spe(nabla_w[0].shape, delta_nabla_w[0].shape)
            # spe(nabla_f[0].shape, delta_nabla_f[0].shape)

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
        conv_id, fc_id = 0, 0
        for i, layer in enumerate(self.layers_avalible):

            if layer.type == layer.CONV:
                layer.weights = self.filters[conv_id].transpose(0, 3, 1, 2)
                layer.biases = self.filters_biases[conv_id].transpose(1, 0)
                conv_id += 1

            if layer.type == layer.FC:
                layer.weights = self.weights[fc_id]
                layer.biases = self.biases[fc_id]
                fc_id += 1

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

