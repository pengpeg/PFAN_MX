# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 7:52
# @Author  : Chen
# @File    : lr_scheduler.py
# @Software: PyCharm
import mxnet as mx

class PFANScheduler(mx.lr_scheduler.LRScheduler):
    def __init__(self, base_lr=0.01, max_iter=100, alpha=10, beta=0.75):
        super(PFANScheduler, self).__init__(base_lr)
        self.factor_p = 1.0 / max_iter
        self.alpha = alpha
        self.beta = beta

    def __call__(self, num_update):
        p = self.factor_p * num_update
        return self.base_lr / (1 + self.alpha * p) ** self.beta

class INVScheduler(mx.lr_scheduler.LRScheduler):
    def __init__(self, gamma=0.001, decay_rate=0.75, base_lr=0.01):
        super(INVScheduler, self).__init__(base_lr)
        self.gamma = gamma
        self.decay_rate = decay_rate

    def __call__(self, num_update):
        lr = self.base_lr * (1 + self.gamma * num_update) ** (-self.decay_rate)
        return lr





