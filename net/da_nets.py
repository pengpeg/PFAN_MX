# -*- coding: utf-8 -*-
# @Time    : 2020/2/10 13:53
# @Author  : Chen
# @File    : da_nets.py
# @Software: PyCharm
import mxnet as mx
from mxnet.gluon import nn
import net.backbone as backbone
from .grad_reverse import grad_reverse

class PFANet(nn.HybridBlock):
    def __init__(self, base_net='AlexNet', use_bottleneck=True, bottleneck_dim=256, class_num=31,
                 ctx=None):
        super(PFANet, self).__init__()
        if not ctx:
            ctx = mx.cpu()
        with self.name_scope():
            self.generator_layer = backbone.network_dict[base_net](ctx)
            self.bottleneck_layer = nn.HybridSequential()
            self.bottleneck_layer.add(nn.Dense(bottleneck_dim))
            self.classifier_layer = nn.HybridSequential()
            self.classifier_layer.add(nn.Dense(class_num))
            self.discriminator_layer = nn.HybridSequential()
            self.discriminator_layer.add(nn.Dense(1024), nn.Activation('relu'), nn.Dropout(0.5))
            self.discriminator_layer.add(nn.Dense(1024), nn.Activation('relu'), nn.Dropout(0.5))
            self.discriminator_layer.add(nn.Dense(2))


        self.bottleneck_layer.initialize(ctx=ctx)
        self.classifier_layer.initialize(ctx=ctx)
        self.discriminator_layer.initialize(ctx=ctx)
        self.generator_layer.collect_params().setattr('lr_mult', 0.1)
        self.use_bottleneck = use_bottleneck

    def hybrid_forward(self, F, x, lamb=0.1):
        features = self.generator_layer(x)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)

        x = grad_reverse(features, lamb)
        outputs_dis = self.discriminator_layer(x)
        outputs_cls = self.classifier_layer(features) / 1.8

        return features, outputs_dis, outputs_cls

