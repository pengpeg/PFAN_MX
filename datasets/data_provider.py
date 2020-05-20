# -*- coding: utf-8 -*-
# @Time    : 2020/2/11 9:49
# @Author  : Chen
# @File    : data_provider.py
# @Software: PyCharm
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data import DataLoader
from .datasets import ImageFolderDataset

def load_images(images_file_path, batch_size, resize_size=256, is_train=True, crop_size=224, is_cen=False,
                sampler=None, pseudo_labels=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if not is_train:
        transformer = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ])
        shuffle = False
        last_bacth = 'keep'
    else:
        if is_cen:
            transformer = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(crop_size),
                transforms.RandomFlipLeftRight(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transformer = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
                transforms.RandomFlipLeftRight(),
                transforms.ToTensor(),
                normalize
            ])
        shuffle = False if sampler is not None else True
        last_bacth = 'keep'

    imageset = ImageFolderDataset(images_file_path, pseudo_labels=pseudo_labels)
    data_loader = DataLoader(
        dataset=imageset.transform_first(transformer),
        shuffle=shuffle,
        batch_size=batch_size,
        last_batch=last_bacth,
        sampler=sampler,
        num_workers=0
    )

    return data_loader



