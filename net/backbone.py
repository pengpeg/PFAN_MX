# -*- coding: utf-8 -*-
# @Time    : 2020/2/10 14:16
# @Author  : Chen
# @File    : backbone.py
# @Software: PyCharm
from mxnet.gluon import nn
import gluoncv as gcv

class AlexNetFc(nn.HybridBlock):
    def __init__(self, ctx):
        super(AlexNetFc, self).__init__()
        model_alexnet = gcv.model_zoo.get_model(name='AlexNet', pretrained=True, ctx=ctx)
        with self.name_scope():
            self.features = model_alexnet.features

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        return x

class ResNet50Fc(nn.HybridBlock):
    def __init__(self, ctx):
        super(ResNet50Fc, self).__init__()
        model_resnet50 = gcv.model_zoo.get_model(name='ResNet50_v1', pretrained=True, ctx=ctx)
        with self.name_scope():
            self.features = model_resnet50.features

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        return x

network_dict = {"AlexNet": AlexNetFc,
                "ResNet50": ResNet50Fc}


