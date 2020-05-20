# -*- coding: utf-8 -*-
# @Time    : 2020/4/19 16:44
# @Author  : Chen
# @File    : loss.py
# @Software: PyCharm
from mxnet.gluon.loss import Loss
from mxnet import autograd
import numpy as np

class ClsTripletLoss(Loss):

    def __init__(self, margin=1.0, weight=1., batch_axis=0, **kwargs):
        super(ClsTripletLoss, self).__init__(weight, batch_axis, **kwargs)
        self._margin = margin

    def hybrid_forward(self, F, features, labels):
        """
        根据triplet loss修改，同类样本间的距离小于一类一个margin.
        此为第二种修改方式：统计所有样本组成的样本对
        """
        num_p = (labels.expand_dims(axis=1) == labels.expand_dims(axis=0)).sum().astype(np.float32) - 128
        num_n = (labels.expand_dims(axis=1) != labels.expand_dims(axis=0)).sum().astype(np.float32)
        with autograd.pause():
            w_same = (labels.expand_dims(axis=1) == labels.expand_dims(axis=0))
            w_same = w_same - F.diag(F.diag(w_same))
            w_diff = (labels.expand_dims(axis=1) != labels.expand_dims(axis=0))
            # w_ij: 同类为1，不同为-1, i==j为0
            w = w_same - w_diff
            # w_ijk: ij同类，jk异类为1，其他为0
            w = (w.expand_dims(axis=2) - w.expand_dims(axis=0) - 1).relu()
            w = w.astype(np.float32)

        distance = ((features.expand_dims(axis=1) - features.expand_dims(axis=0)) ** 2).sum(axis=-1)
        # loss_ijk = d_ij - d_jk
        loss = (distance.expand_dims(axis=2) - distance.expand_dims(axis=0) + self._margin).relu()
        loss = w * loss
        loss = loss.sum() / w.sum()
        return loss
