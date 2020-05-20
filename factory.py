# -*- coding: utf-8 -*-
# @Time    : 2020/3/7 17:48
# @Author  : Chen
# @File    : factory.py
# @Software: PyCharm
from net.da_nets import PFANet
from datasets.data_provider import load_images
from datasets.datasets import IdxSampler
import math, os, time
import numpy as np
import mxnet as mx
from mxnet import autograd, gluon
from lr_scheduler.lr_scheduler import PFANScheduler
from loss.loss import ClsTripletLoss

class Factory:
    def __init__(self, logger, data_dir, checkpoint_dir, base_net='AlexNet', batch_size=128, source='amazon',
                 target='webcam', num_classes=31, u=0.3, base_lr=0.01, alpha=10, beta=0.75, max_iter=100, max_step=10,
                 interval=5):
        self.context = mx.gpu()
        self.logger = logger
        self.checkpoint_dir = checkpoint_dir
        self.u = u
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.classes = np.arange(self.num_classes)
        self.base_lr = base_lr
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.max_step = max_step
        self.interval = interval
        self.source = source
        self.target = target
        self.source_file = os.path.join(data_dir, source, 'images')
        self.target_file = os.path.join(data_dir, target, 'images')
        self.base_net = base_net
        self.net = PFANet(base_net=base_net, class_num=self.num_classes, ctx=self.context)
        self.set_lamb()
        self.K = 10

    def set_lamb(self):
        p = np.arange(self.max_iter) + 1.0
        p = p / (self.max_iter)
        self._lamb = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1

    def set_lr(self, num_iter):
        p = np.arange(self.max_iter)
        self._lr = self.base_lr / pow(1 + 0.001 * p, 0.75)

    def transfer_learning(self):
        # Stage 1: Initialize G and F using D_s, output: model_0
        model_pretained_prefix = '{}/PFAN_{}_pretrain_on_{}.params'.format(self.checkpoint_dir, self.base_net, self.source)
        if os.path.exists(model_pretained_prefix):
            self.net.load_parameters(model_pretained_prefix)
        else:
            self.logger.debug('To initialize generator and classifier layer by fine-tune.')
            self.fine_tune()
            self.logger.debug('Initialization finished!')
            self.net.save_parameters(model_pretained_prefix)

        # Stage 2:
        converge = False
        m = 0
        while not converge and m < self.max_step:
            # Run the EHTS based on model_m−1, output: D^_t
            # Calculate the initial global prototypes
            threshold = 1. / (1 + math.exp(-self.u * (m + 1))) - 0.01
            tic1 = time.time()
            proto_g_s, proto_g_t, pseudo_labels, idxes_selected, acc_pseudo, acc_test = self.process_EHTS(threshold)
            tic2 = time.time()
            log = 'Step %d: threshold: %.4f,pseudo number: %d, pseudo accuracy: %.4f, test accuracy: %.4f' % \
                  (m, threshold, idxes_selected.shape[0], acc_pseudo, acc_test)
            self.logger.info(log)
            if idxes_selected.shape[0] < self.batch_size:
                self.logger.debug('The number of pseudo-labeled sample shuld be larger than batch size!')
                break
            # Adaptive Prototype Alignment
            tic3 = time.time()
            self.process_APA(pseudo_labels, idxes_selected, proto_g_s, proto_g_t)
            tic4 = time.time()
            self.logger.info('Step %d: the EHTS process consume %.1f, the APA process consume %.1f.' % (m, tic2 - tic1, tic4 - tic3))
            m = m + 1

    def process_EHTS(self, threshold):
        loader_test_s = load_images(self.source_file, batch_size=self.batch_size, is_train=False)
        loader_test_t = load_images(self.target_file, batch_size=self.batch_size, is_train=False)
        features_s, preds_s, labels_s, idxes_s, accuracy_s = self.forward(loader_test_s)
        features_t, preds_t, labels_t, idxes_t, accuracy_t = self.forward(loader_test_t)
        num_samples_s = features_s.shape[0]
        num_samples_t = features_t.shape[0]

        left_factor_s = self.classes.reshape((self.num_classes, 1)) == labels_s.reshape((1, num_samples_s))  # left_factor: num_classes * num_samples_s
        proto_s_g = np.dot(left_factor_s, features_s) / (left_factor_s.sum(axis=1).reshape((self.num_classes, 1)) + 0.0001)  # num_classes * dims

        # assign pseudo-labels to target sample
        a_s = self.pseudo_by_density(features_t)

        norm_features_t = (features_t ** 2).sum(axis=1).reshape(num_samples_t, 1)
        norm_proto_s_g = (proto_s_g ** 2).sum(axis=1).reshape(self.num_classes, 1)
        similarity_cosine = np.dot(features_t, proto_s_g.T) / (np.sqrt(np.dot(norm_features_t, norm_proto_s_g.T)) + 0.0001)
        scores_t = similarity_cosine.max(axis=1)
        pseudo_labels = similarity_cosine.argmax(axis=1)
        idxes_selected = np.where(scores_t > threshold)[0]
        features_t_selected = features_t[idxes_selected]
        pseudo_labels_selected = pseudo_labels[idxes_selected]

        left_factor_t = self.classes.reshape((self.num_classes, 1)) == pseudo_labels_selected.reshape((1, idxes_selected.shape[0]))
        proto_t_selected = np.dot(left_factor_t, features_t_selected) / (left_factor_t.sum(axis=1).reshape((self.num_classes, 1)) + 0.0001)

        self.logger.info('number of each class: ' + str(left_factor_t.sum(axis=1)))
        acc_pseudo = self.calPseudoLabelAcc(labels_t, idxes_t, pseudo_labels, idxes_selected)

        return proto_s_g, proto_t_selected, pseudo_labels, idxes_selected, acc_pseudo, accuracy_t

    def process_APA(self, pseudo_labels, idxes_selected, proto_s_g, proto_t_g):
        lr_scheduler = PFANScheduler(base_lr=self.base_lr, max_iter=self.max_iter, alpha=self.alpha, beta=self.beta)
        optimizer = mx.optimizer.SGD(learning_rate=self.base_lr, momentum=0.9, wd=0.0005, lr_scheduler=lr_scheduler)
        trainer = mx.gluon.Trainer(self.net.collect_params(), optimizer=optimizer)
        metric_acc_cls_train = mx.metric.Accuracy()
        metric_acc_dis_train = mx.metric.Accuracy()
        softmax_cross_entropy_F = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        l2_apa = mx.gluon.loss.L2Loss()
        softmax_cross_entropy_D = mx.gluon.loss.SoftmaxCrossEntropyLoss()

        sampler_train_t = IdxSampler(indices_selected=idxes_selected)
        loader_train_s = load_images(self.source_file, batch_size=self.batch_size)
        loader_train_t = load_images(self.target_file, batch_size=self.batch_size, sampler=sampler_train_t,
                                     pseudo_labels=pseudo_labels)
        loader_test_t = load_images(self.target_file, batch_size=self.batch_size, is_train=False)
        iter_s = iter(loader_train_s)
        iter_t = iter(loader_train_t)

        # proto_g_initial = mx.nd.concatenate([proto_s_g, proto_t_g]) # The initial global prototypes
        proto_s_g_I_1 = mx.nd.array(proto_s_g).as_in_context(self.context)  # The I-1 accumulated global prototypes
        proto_t_g_I_1 = mx.nd.array(proto_t_g).as_in_context(self.context)
        proto_s_l_I_1_ = 0  # The I-1 accumulated local prototypes of all samples
        proto_t_l_I_1_ = 0
        label_dis = mx.nd.ones(shape=(self.batch_size * 2,), ctx=self.context)
        label_dis[:self.batch_size] = 0
        I = 1
        while I < self.max_iter + 1:
            tic = time.time()
            try:
                batch_s = next(iter_s)
            except StopIteration:
                iter_s = iter(loader_train_s)
                batch_s = next(iter_s)
            try:
                batch_t = next(iter_t)
            except StopIteration:
                iter_t = iter(loader_train_t)
                batch_t = next(iter_t)
            if batch_s[0].shape[0] < self.batch_size or batch_t[0].shape[0] < self.batch_size:
                continue
            data_s = batch_s[0].as_in_context(self.context)
            label_s = batch_s[1].as_in_context(self.context)
            data_t = batch_t[0].as_in_context(self.context)
            pseudo_label_t = batch_t[3].as_in_context(self.context)
            data = mx.nd.concatenate([data_s, data_t])
            # label = mx.nd.concatenate([label_s, pseudo_label_t])

            left_factor_s = np.arange(self.num_classes).reshape((self.num_classes, 1)) == label_s.asnumpy().reshape(
                (1, label_s.shape[0]))
            left_factor_t = np.arange(self.num_classes).reshape(
                (self.num_classes, 1)) == pseudo_label_t.asnumpy().reshape((1, pseudo_label_t.shape[0]))
            right_division_s = left_factor_s.sum(axis=1).reshape((self.num_classes, 1)) + 0.0001
            right_division_t = left_factor_t.sum(axis=1).reshape((self.num_classes, 1)) + 0.0001
            # right_division_s[np.where(right_division_s == 0)] = 1
            # right_division_t[np.where(right_division_t == 0)] = 1
            left_factor_s = mx.nd.array(left_factor_s).as_in_context(self.context)
            left_factor_t = mx.nd.array(left_factor_t).as_in_context(self.context)
            right_division_s = mx.nd.array(right_division_s).as_in_context(
                self.context)  # some class sample number maybe be zero
            right_division_t = mx.nd.array(right_division_t).as_in_context(self.context)
            # APA process with mini-batch from training set (S and pseudo-labeled target sample)
            with autograd.record():
                features, outputs_dis, outputs_cls = self.net(data)
                outputs_cls_s = mx.nd.slice(outputs_cls, 0, self.batch_size)
                features_s = mx.nd.slice(features, 0, self.batch_size)
                features_t = mx.nd.slice(features, self.batch_size, 2 * self.batch_size)

                proto_s_l_I = mx.nd.dot(left_factor_s, features_s) / right_division_s  # The I source local prototypes
                proto_t_l_I = mx.nd.dot(left_factor_t, features_t) / right_division_t  # The I target local prototypes
                proto_s_l_I_ = ((I - 1) * proto_s_l_I_1_ + proto_s_l_I) / I  # The I accumulated local prototypes
                proto_t_l_I_ = ((I - 1) * proto_t_l_I_1_ + proto_t_l_I) / I
                with autograd.pause():
                    norm_s = mx.nd.sqrt((proto_s_l_I_ ** 2).sum(axis=1) * (proto_s_g_I_1 ** 2).sum(axis=1)) + 0.0001
                    norm_t = mx.nd.sqrt((proto_t_l_I_ ** 2).sum(axis=1) * (proto_t_g_I_1 ** 2).sum(axis=1)) + 0.0001
                    p_s = ((proto_s_l_I_ * proto_s_g_I_1).sum(axis=1) / norm_s).reshape(shape=(self.num_classes, 1))
                    p_t = ((proto_t_l_I_ * proto_t_g_I_1).sum(axis=1) / norm_t).reshape(shape=(self.num_classes, 1))
                # proto_s_g_I = p_s ** 2 * proto_s_l_I_ + (1 - p_s ** 2) * proto_s_g_I_1  # The I accumulated global prototypes
                # proto_t_g_I = p_t ** 2 * proto_t_l_I_ + (1 - p_t ** 2) * proto_t_g_I_1
                proto_s_g_I = 0.3 * proto_s_l_I_ + (1 - 0.3) * proto_s_g_I_1
                proto_t_g_I = 0.3 * proto_t_l_I_ + (1 - 0.3) * proto_t_g_I_1

                loss_cls = softmax_cross_entropy_F(outputs_cls_s, label_s).mean()
                loss_dis = softmax_cross_entropy_D(outputs_dis, label_dis).mean()
                loss_apa = l2_apa(proto_t_g_I, proto_s_g_I).mean()
                loss_all = loss_cls + 1.0 * loss_dis + 1.0 * loss_apa
                loss_all.backward()
            trainer.step(1, ignore_stale_grad=True)
            metric_acc_cls_train.update(label_s, outputs_cls_s)
            metric_acc_dis_train.update(label_dis, outputs_dis)
            proto_s_g_I_1 = proto_s_g_I.copy()
            proto_t_g_I_1 = proto_t_g_I.copy()
            proto_s_l_I_1_ = proto_s_l_I_.copy()
            proto_t_l_I_1_ = proto_t_l_I_.copy()

            if I % self.interval == 0:
                _, acc_cls_train = metric_acc_cls_train.get()
                _, acc_dis_train = metric_acc_dis_train.get()
                metric_acc_cls_train.reset()
                metric_acc_dis_train.reset()
                log = 'Training iter: %d, Train cls acc: %.4f, dis acc: %.4f, cls loss: %.4f, dis loss: %.4f, apa loss: %.4f, lr: %.2E, time: %.2f/iter' % (
                    I, acc_cls_train, acc_dis_train, loss_cls.mean().asscalar(), loss_dis.mean().asscalar(),
                    loss_apa.mean().asscalar(), trainer.learning_rate, time.time() - tic)
                self.logger.debug(log)

            if I % self.eval_interval == 0:
                _, _, _, _, acc_test = self.forward(loader_test_t)
                # acces_test.append(acc_test)
                log = 'Iter: %d, Test acc: %.4f' % (I, acc_test)
                self.logger.debug(log)
            I += 1

    def fine_tune(self):
        steps = [1500, 2000]
        lr = 0.01
        max_iter = 2000
        batch_size = 32
        p = 0.01 / max_iter
        lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, base_lr=lr, factor=0.1)
        optimizer = mx.optimizer.SGD(learning_rate=lr, momentum=0.9, wd=0.0005, lr_scheduler=lr_scheduler)
        trainer = mx.gluon.Trainer(self.net.collect_params(), optimizer=optimizer)

        metric_acc = mx.metric.Accuracy()
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
        cls_triplet_loss = ClsTripletLoss(margin=100)

        loader_train = load_images(self.source_file, batch_size=batch_size)
        loader_test = load_images(self.target_file, batch_size=batch_size, is_train=False)
        train_batch_per_epoch = len(loader_train)

        iter_s = iter(loader_train)
        num_iter = 1
        max_iter = max_iter - max_iter % train_batch_per_epoch
        while num_iter < max_iter + 1:
            try:
                batch = next(iter_s)
            except StopIteration:
                iter_s = iter(loader_train)
                batch = next(iter_s)
            if batch[0].shape[0] < batch_size:
                continue
            data = batch[0].as_in_context(self.context)
            label = batch[1].as_in_context(self.context)
            with autograd.record():
                features, _, outputs_cls = self.net(data)
                loss_cls = softmax_cross_entropy(outputs_cls, label).mean()
                loss_clstriplet = cls_triplet_loss(features, label).mean()
                loss_all = loss_cls  # + (0.001 + p * num_iter) * loss_clstriplet
                loss_all.backward()
            trainer.step(batch_size=1, ignore_stale_grad=True)
            metric_acc.update(labels=label, preds=outputs_cls)
            if num_iter % self.interval == 0:
                _, acc_train = metric_acc.get()
                metric_acc.reset()
                log = 'Training iter: %d, Train acc: %.4f, loss: %.4f, triplet loss: %.4f, lr: %3E' % \
                      (num_iter, acc_train, loss_cls.asscalar(), loss_clstriplet.asscalar(), trainer.learning_rate)
                self.logger.debug(log)

            if num_iter % train_batch_per_epoch == 0:
                _, _, _, _, acc_test = self.forward(loader_test)
                # acces_test.append(acc_test)
                log = 'Iter: %d, Test acc: %.4f' % (num_iter, acc_test)
                self.logger.debug(log)
            num_iter += 1

    def forward(self, loader_data):
        features_all, preds_all, labels_all, idxes_all = [], [], [], []
        metric_acc = mx.metric.Accuracy()
        for i, batch in enumerate(loader_data):
            data, labels, idxes = batch[0], batch[1], batch[2]
            data = data.as_in_context(self.context)
            features, _, outputs_cls = self.net(data)
            metric_acc.update(labels, outputs_cls)
            features_all.append(features.asnumpy())
            preds_all.append(mx.nd.softmax(outputs_cls).asnumpy())
            labels_all.append(labels.asnumpy())
            idxes_all.append(idxes.asnumpy())
        features = np.concatenate(features_all)
        preds = np.concatenate(preds_all)
        labels = np.concatenate(labels_all)
        idxes = np.concatenate(idxes_all)

        indice_sorted = np.argsort(idxes)
        idxes = idxes[indice_sorted]
        if not (idxes == np.arange(idxes.shape[0])).all():
            print('The sample idx exits error!')
            return
        features = features[indice_sorted]
        # outputs_cls = outputs_cls[indice_sorted]
        labels = labels[indice_sorted]

        _, accuracy = metric_acc.get()

        return features, preds, labels, idxes, accuracy

    def calPseudoLabelAcc(self, labels, idxes, pseudo_labels, idxes_selected):
        assert labels.shape == pseudo_labels.shape
        assert (idxes == np.arange(idxes.shape[0])).all()

        correct = (labels[idxes_selected] == pseudo_labels[idxes_selected]).sum()
        acc = float(correct) / idxes_selected.shape[0]
        return acc

    def pseudo_by_density(self, features_t):
        norm_t = np.expand_dims(np.sqrt((features_t ** 2).sum(axis=1)), axis=1)
        similarity_cosine = np.dot(features_t, features_t.T) / np.dot(norm_t, norm_t.T)
        s_sorted = np.sort(similarity_cosine)
        i_sorted = np.argsort(similarity_cosine)
        a_s = s_sorted[:, -self.K-1:-1].sum(axis=1) / self.K
        return a_s

