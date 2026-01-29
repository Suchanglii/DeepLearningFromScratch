# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def load_params(self, params_dict):
        for key in ('W1', 'b1', 'W2', 'b2'):
            self.params[key] = params_dict[key]

        # 注意：更新了 params 后，需要重新初始化 Affine 层，因为层里持有的是权重的引用
        self.layers['Affine1'].W = self.params['W1']
        self.layers['Affine1'].b = self.params['b1']
        self.layers['Affine2'].W = self.params['W2']
        self.layers['Affine2'].b = self.params['b2']

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        # 这里调用一次后，返回的相当于没有经过SoftmaxWithLoss（）的结果，就是Affine2算完的结果
        return x
        
    # 调用后得到最终输出结果
    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        # 输出层先进行一次反向传播
        dout = self.lastLayer.backward(dout)
        # 考虑：是否可以不经过softmax层，从新定义一个原样输出层，将结果直接输出，然后算损失，直接反向0传播
        # 不可以直接这样做。数学原因：反向传播的核心是链式法则。SoftmaxWithLoss 层在求导时有一个非常优雅的特性：
        # 它的梯度结果简单地就是 $y - t$（预测值减去真实值）。如果你跳过Softmax直接算Loss的导数，计算会变得异常复杂，
        # 甚至失去概率意义上的梯度下降方向。
        # 如果你想看“原样输出”，只需调用 predict(x)即可，但在训练（计算梯度）时，Softmax 和Loss
        # 必须成对出现才能保证反向传播公式的简洁和正确。
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        # SoftmaxWithLoss相对于每一层的权重、偏置的梯度
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
