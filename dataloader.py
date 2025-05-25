# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:38:25 2020

@author: Wu Yichen
"""

from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
import pickle


import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

import torch
import torch.nn.functional as F
from torch.autograd import Variable as V
import torchvision.transforms as transforms
import random


def uniform_mix_C(mixing_ratio, num_classes):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)

def flip_labels_C(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C

def flip_labels_C_two(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_prob / 2
    return C

def oneh2img(X_train):
    # 第一步：每行元素后补7个0，使其变为 (1000, 90)
    pad = 90 - X_train.shape[1]
    assert  pad > 0
    X_train_padded = np.pad(X_train, ((0, 0), (0, pad)), mode='constant', constant_values=0)
    # 第二步：复制每行元素10份，使其变为 (1000, 30, 30)
    X_train_reshaped = np.repeat(X_train_padded, 10, axis=1).reshape(-1, 30, 30)
    # 第三步：每个元素复制3份，使其变为 (1000, 3, 30, 30)
    X_train_final = np.repeat(X_train_reshaped[:, np.newaxis, :, :], 3, axis=1)
    return X_train_final  # 输出: (1000, 3, 30, 30)


def random_horizontal_flip(matrix, ratio = 0.5):
    # 以 50% 的概率进行水平翻转
    if random.random() < ratio:
        # 水平翻转矩阵
        flipped_matrix = matrix[:, ::-1, :].copy()
        return flipped_matrix
    return matrix


class CIFAR10(data.Dataset):
    def __init__(self, root='', train=True, meta=True, epochs_wp=80,
                 corruption_prob=0, corruption_type='sy', transform=None, target_transform=None,
                 num_classes=6, seed=1, meta_name='meta_data_random.npz', dataset='ids18'):
        self.count = 0
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.meta = meta
        self.epochs_wp = epochs_wp
        self.corruption_prob = corruption_prob
        self.corruption_type = corruption_type
        self.num_classes = num_classes
        self.num_meta = 0


        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_data_fea = []
            self.train_labels = []
            self.train_coarse_labels = []
            self.train_labels_true = []
            self.soft_label_1 = []
            self.soft_label_2 = []
            self.fhat_label_1 = []

            if self.meta is True:
                data = np.load(os.path.join(self.root, meta_name))
                train_data_fea = data['X_meta']
                train_data = data['X_meta']
                train_labels_true = data['y_meta']
                noise_label = train_labels_true.copy()
                self.num_meta = len(train_labels_true)

                self.meta_data_fea = train_data_fea
                self.meta_labels_true = train_labels_true

            else:
                data = np.load(os.path.join(self.root, 'noise_data.npz'))
                if dataset == 'ids18':
                    train_data = data['X_train']
                    train_data_fea = data['X_train']
                    train_labels_true = data['y_train']
                    self.random_indices = list(range(len(train_labels_true)))
                    data_noilabel = np.load(os.path.join(self.root, 'noise_' + self.corruption_type + '_labels.npz'))
                    noise_label = data_noilabel['y' + str(int(self.corruption_prob * 10))]

            train_data = oneh2img(train_data)

            # self.train_data = train_data.transpose((0, 2, 3, 1)) # convert to HWC
            self.train_data_fea = train_data_fea
            self.train_data = train_data
            self.train_labels_true = train_labels_true
            self.train_labels = noise_label
            self.soft_label_1 = list(np.zeros((len(self.train_data), self.num_classes), dtype=np.float32))
            self.soft_label_2 = list(np.zeros((len(self.train_data)), dtype=np.float32))
            self.fhat_label = list(np.zeros((len(self.train_data), self.num_classes), dtype=np.float32))
            self.fhat_label_1 = list(np.zeros((len(self.train_data), self.num_classes), dtype=np.float32))
            self.prediction = np.zeros((len(self.train_data), 10, self.num_classes), dtype=np.float32)

        else:
            data = np.load(os.path.join(self.root, 'noise_data.npz'))
            test_data = data['X_test']
            test_data_fea = data['X_test']
            test_labels_true = data['y_test']

            test_data = oneh2img(test_data)

            # self.test_data = test_data.transpose((0, 2, 3, 1)) # convert to HWC
            self.test_data = test_data
            self.test_data_fea = test_data_fea
            self.test_labels = test_labels_true


    def label_update(self, fhat_label_1, fhat_label=0, soft_label_1=0, soft_label_2=0):
        self.count += 1
        # While updating the noisy label y_i by the probability s, we used the average output probability of the network of the past 10 epochs as s.
        idx = (self.count - 1) % 10#10 #10
        self.prediction[:, idx] = fhat_label_1
        self.fhat_label_1 = fhat_label_1


        if self.count == self.epochs_wp - 1: #79
            self.soft_label_1 = self.prediction.mean(axis=1)

            mask = np.ones(self.prediction.shape[1], dtype=bool)
            mask[idx] = False
            soft_label_2_tmp = self.prediction[:, mask, :].mean(axis=1)
            self.soft_label_2 = list(np.argmax(soft_label_2_tmp, axis=1).astype(np.int64))

        if self.count > self.epochs_wp - 1:
            self.soft_label_1 = soft_label_1
            self.soft_label_2 = list(soft_label_2.astype(np.int64))
            self.fhat_label = fhat_label
            #self.soft_labels = list(np.argmax(self.soft_labels, axis=1).astype(np.int64))

    def set_prediction(self, load_pre, epoch, epoch_wp):
        # self.prediction = load_pre
        self.prediction = load_pre[self.random_indices,:,:]
        # self.prediction = load_pre[:2000]
        self.count = epoch

        idx = (self.count - 1) % 10  # 10 #10
        # self.fhat_label_1 = np.argmax(self.prediction[:, idx], axis=1).astype(np.int64)
        self.fhat_label_1 = self.prediction[:, idx]
        self.soft_label_1 = self.prediction.mean(axis=1)

        mask = np.ones(self.prediction.shape[1], dtype=bool)
        mask[idx] = False
        soft_label_2_tmp = self.prediction[:, mask, :].mean(axis=1)
        self.soft_label_2 = list(np.argmax(soft_label_2_tmp, axis=1).astype(np.int64))

        self.count = epoch_wp - 1



    def update_meta(self, exped_dataset):
        if self.train:
            if self.meta is True:
                # add expend data
                if len(exped_dataset[0]) != 0 :
                    exp_train_data = exped_dataset[0]
                    exp_train_data_fea = exped_dataset[0].copy()
                    exp_train_labels = exped_dataset[1]
                    exp_train_labels_true = exped_dataset[2]

                    exp_train_data = oneh2img(exp_train_data)
                    ori_train_data = oneh2img(self.meta_data_fea)

                    # self.train_data = train_data.transpose((0, 2, 3, 1)) # convert to HWC
                    # self.train_data_fea = np.concatenate((self.train_data_fea, exp_train_data_fea), axis=0)
                    # self.train_data = np.concatenate((self.train_data, exp_train_data), axis=0)
                    # self.train_labels_true = np.concatenate((self.train_labels_true, exp_train_labels_true), axis=0)
                    # self.train_labels = np.concatenate((self.train_labels, exp_train_labels), axis=0)

                    self.train_data_fea = np.concatenate((self.meta_data_fea, exp_train_data_fea), axis=0)
                    self.train_data = np.concatenate((ori_train_data, exp_train_data), axis=0)
                    self.train_labels_true = np.concatenate((self.meta_labels_true, exp_train_labels_true), axis=0)
                    self.train_labels = np.concatenate((self.meta_labels_true, exp_train_labels), axis=0)

                    self.num_meta = len(self.train_data)

            else:
                raise RuntimeError('Training Dataset can not be expended')
        else:
            raise RuntimeError('Testing Dataset can not be expended')


    def __getitem__(self, index):
        if self.train:
            if self.meta:
               img, fea, target, target_true= self.train_data[index], self.train_data_fea[index], self.train_labels[index],self.train_labels_true[index]
            else:
               img, fea, target, target_true= self.train_data[index], self.train_data_fea[index], self.train_labels[index], self.train_labels_true[index]
               fhat_label = self.fhat_label[index]
               fhat_label1 = self.fhat_label_1[index]
               soft_label1 = self.soft_label_1[index]
               soft_label2 = self.soft_label_2[index]

        else:
            img, fea, target = self.test_data[index], self.test_data_fea[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        # if self.transform is not None:
        #     img = self.transform(img)

        if self.transform == 0:
            img = torch.tensor(img, dtype=torch.float32)
        elif self.transform == 1:
            img = random_horizontal_flip(img)
            img = torch.tensor(img, dtype=torch.float32)
        elif self.transform == 20:
            img1 = torch.tensor(img, dtype=torch.float32)
            img2 = torch.tensor(img, dtype=torch.float32)
            img  = (img1, img2)
        elif self.transform == 21:
            img1 = random_horizontal_flip(img)
            img1 = torch.tensor(img1, dtype=torch.float32)
            img2 = random_horizontal_flip(img)
            img2 = torch.tensor(img2, dtype=torch.float32)
            img = (img1, img2)


        if self.target_transform is not None:
            target = self.target_transform(target)

        target = np.int64(target)

        if self.train :
           fea = torch.tensor(fea, dtype=torch.float32)
           if self.meta:
               return img, fea, target, target_true
           else:
               target_true = np.int64(target_true)
               return img, fea, target, target_true, fhat_label, fhat_label1, soft_label1, soft_label2, index
        else:
           return img, fea, target

        
    def __len__(self):
        if self.train:
            if self.meta is True:
                return self.num_meta
            else:
                return len(self.train_data)
        else:
            return len(self.test_data)




class CIFAR100(CIFAR10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
