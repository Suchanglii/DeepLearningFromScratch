# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    # 损失函数针对的对象是输入x经过神经网络后输出结果y与真实结果t之间的差异
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        print(f'y = {y}')
        # x一变，预测结果y就会变，y变又导致损失函数结果变化，损失函数变化，就能算出每个dloss/dx
        loss = cross_entropy_error(y, t) # 交叉熵
        print(f'loss = {loss};loss.type = {type(loss)};loss.shape = {loss.shape}')
        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()
print(f'net.W = {net.W}')

p = net.predict(x)
print(f'p = {p}')
print(f'np.argmax(p) = {np.argmax(p)}')

# print(f'net.loss() = {net.loss(x, t)}')

def f(W):
     return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)

# f = lambda w: net.loss(x, t)
# dW = numerical_gradient(f, net.W)
#
# print(dW)
