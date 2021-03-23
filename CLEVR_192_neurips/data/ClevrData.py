
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys

import random
import torch
from skimage import io
import numpy  as np
import PIL
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from torchvision import datasets
import glob
import json
from utils import *


import lmdb
import pickle
from collections import namedtuple

shape_dict = {
    "cube": 0,
    "cylinder": 1,
    "sphere": 2
}

color_dict  = {
    "gray": 0,
    "red": 1,
    "blue": 2,
    "green": 3,
    "brown": 4,
    "purple": 5,
    "cyan": 6,
    "yellow": 7
}

"""
cube:
    grey 0
    red 1
    blue 2
    green 3
    brown 4
    purple 5
    cyan 6
    yellow 7
cylinder:
    grey 8
    red 9
    blue 10
    green 11
    brown 12
    purple 13
    cyan 14
    yellow 15
sphere:
    grey 16
    red 17
    blue 18
    green 19
    brown 20
    purple 21
    cyan 22
    yellow 23


"""


class ClevrDataset(Dataset):
    def __init__(self, data_dir, imsize, split='train', transform=None):

        self.transform = transform
        self.imsize = imsize
        self.split = split
        self.data = []
        self.data_dir = data_dir
        self.split_dir = os.path.join(data_dir, split)
        self.img_dir = os.path.join(self.split_dir, "images")
        self.scene_dir = os.path.join(self.split_dir, "scenes")
        self.max_objects = 2

        self.filenames = self.load_filenames()

    def get_img(self, img_path):
        #print(img_path)
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def load_bboxes(self):
        bbox_path = os.path.join(self.split_dir, 'bboxes.pickle')
        with open(bbox_path, "rb") as f:
            bboxes = pickle.load(f)
            bboxes = np.array(bboxes)
        return bboxes

    def load_labels(self):
        label_path = os.path.join(self.split_dir, 'labels.pickle')
        with open(label_path, "rb") as f:
            labels = pickle.load(f)
            labels = np.array(labels)
        return labels

    def load_filenames(self):
        filenames = [filename for filename in glob.glob(self.scene_dir + '/*.json')]
        print('Load scenes from: %s (%d)' % (self.scene_dir, len(filenames)))
        return filenames

    def label_one_hot(self, label, dim):
        labels = torch.from_numpy(label)
        labels = labels.long()
        # remove -1 to enable one-hot converting
        labels[labels < 0] = dim-1
        label_one_hot = torch.FloatTensor(labels.shape[0], dim).fill_(0)
        label_one_hot = label_one_hot.scatter_(1, labels, 1).float()
        return label_one_hot

    def __getitem__(self, index):
        # load image
        key = self.filenames[index]
        with open(key, "rb") as f:
            json_file = json.load(f)

        img_name = key[:18]+"/images"+key[25:-5]+".png"
        if self.split=="test":
            img_name = key[:18]+"images/"+key[25:-5]+".png"
        img = self.get_img(img_name)

        # load bbox#
        bbox = np.zeros((self.max_objects, 4), dtype=np.float32)
        bbox[:] = -1.0
        for idx in range(len(json_file["objects"])):
            bbox[idx, :] = json_file["objects"][idx]["bbox"]
        bbox = bbox / float(self.imsize)

        label = np.zeros(self.max_objects)
        label[:] = -1
        for idx in range(len(json_file["objects"])):
            shape_idx = shape_dict[json_file["objects"][idx]["shape"]]
            color_idx = color_dict[json_file["objects"][idx]["color"]]
            value = None
            if shape_idx==0:
                value = color_idx
            elif shape_idx==1:
                value = 8 + color_idx
            elif shape_idx==2:
                value = 16 + color_idx
            label[idx] = value


        label = self.label_one_hot(np.expand_dims(label, 1), 25)

        return img, label, bbox, key[25:-5]

    def __len__(self):
        return len(self.filenames)

CodeRow = namedtuple('CodeRow', ['q1', 'q2',  'lbls', 'bboxes'])

class ImageFileDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        dirs, filename = os.path.split(path)
        _, class_name = os.path.split(dirs)
        filename = os.path.join(class_name, filename)

        return sample, target, filename

class LMDBDataset(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))

        q1 = torch.from_numpy(row.q1)
        q2 = torch.from_numpy(row.q2)
        lbls = torch.from_numpy(row.lbls)
        bboxes = torch.from_numpy(row.bboxes)
        return q1, q2, lbls, bboxes
