# -*- coding: utf-8 -*-
# @Time    : 2020/3/5 14:07
# @Author  : Chen
# @File    : grad_reverse.py
# @Software: PyCharm
from mxnet.autograd import Function

class GradReverse(Function):
    def __init__(self, lambd):
        super(GradReverse, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return x * 1

    def backward(self, grad_output):
        return -self.lambd * grad_output


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)



