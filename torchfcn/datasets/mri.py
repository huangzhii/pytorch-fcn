#!/usr/bin/env python

import collections
import os.path as osp
import time

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import pylab


def happyprint(string, obj):
    print(string, obj)
    return


class MRIClassSegBase(data.Dataset):

#| Name       | (r,g,b)         |  7-Class mapping   | labels_mono
#|------------|-----------------|--------------------|
#| Car        | (  0,  0,255)   | Object             | 1
#| Road       | (255,  0,  0)   | Road               | 2
#| Mark       | (255,255,  0)   | Road               | 3
#| Building   | (  0,255,  0)   | Building           | 4
#| Sidewalk   | (255,  0,255)   | Road               | 5
#| Tree/Bush  | (  0,255,255)   | Tree/Bush          | 6
#| Pole       | (255,  0,153)   | Sign/Pole          | 7
#| Sign       | (153,  0,255)   | Sign/Pole          | 8
#| Person     | (  0,153,255)   | Object             | 9
#| Wall       | (153,255,  0)   | Building           | 10
#| Sky        | (255,153,  0)   | Sky                | 11
#| Curb       | (  0,255,153)   | Road               | 12
#| Grass/Dirt | (  0,153,153)   | Grass/Dirt         | 13
#| Void       | (  0,  0,  0)   | Void               | -1

    class_names = np.array([
        'CA1',
        'CA2',
        'CA3',
        'DG',
        'SUB',
        'ERC',
        'BA35',
        'BA36',
        'background',
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform

        dataset_dir = self.root
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(
            dataset_dir, 'splits', 'all.txt')
            for did in open(imgsets_file):
                did = did.strip('.pcd\n')
                img_file = osp.join(dataset_dir, 'images/%s.png' % did)
                lbl_file = osp.join(dataset_dir, 'labels/%s.png' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        # img = np.expand_dims(img, axis=3)
        # lbl = np.expand_dims(lbl, axis=3)

        # fake data
        img = np.random.randn(1, 32, 32, 32) + 30
        img = np.array(img, dtype=np.uint8)
        lbl = np.random.randn(32, 32, 32) + 4
        lbl = np.array(lbl, dtype=np.int32)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

        # lbl[lbl == 255] = -1
        # lbl[lbl == 0] = -1
        # lbl = self.lbl_color_transform(lbl)
        # print("plotting")
        # print(np.unique(lbl))
        # imgplot = plt.imshow(lbl)
        # pylab.show()
        # time.sleep(55)    # pause 5.5 seconds
        
        # if self._transform:
        #     return self.transform(img, lbl)
        # else:
        #     return img, lbl


    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        # happyprint("img: ", img.size())
        # happyprint("lbl: ", lbl)
        img = img.numpy()
        # img = img.transpose(1, 2, 0)
        # img += self.mean_bgr
        img = img.astype(np.uint8)
        # img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl


class MRIClassSegValidate(MRIClassSegBase):

    def __init__(self, root, split='train', transform=False):
        super(MRIClassSegValidate, self).__init__(
            root, split=split, transform=transform)
        dataset_dir = self.root
        imgsets_file = osp.join(
            dataset_dir, 'splits', 'train_small.txt')
        for did in open(imgsets_file):
            did = did.strip('.pcd\n')
            img_file = osp.join(dataset_dir, 'images/%s.png' % did)
            lbl_file = osp.join(dataset_dir, 'labels/%s.png' % did)
            self.files['validation'].append({'img': img_file, 'lbl': lbl_file})


class MRIClassSeg(MRIClassSegBase):

    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA

    def __init__(self, root, split='train', transform=False):
        super(MRIClassSeg, self).__init__(
            root, split=split, transform=transform)


# class SBDClassSeg(MRIClassSegBase):

#     # XXX: It must be renamed to benchmark.tar to be extracted.
#     url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'  # NOQA

#     def __init__(self, root, split='train', transform=False):
#         self.root = root
#         self.split = split
#         self._transform = transform

#         dataset_dir = osp.join(self.root, 'VOC/benchmark_RELEASE/dataset')
#         self.files = collections.defaultdict(list)
#         for split in ['train', 'val']:
#             imgsets_file = osp.join(dataset_dir, '%s.txt' % split)
#             for did in open(imgsets_file):
#                 did = did.strip()
#                 img_file = osp.join(dataset_dir, 'img/%s.jpg' % did)
#                 lbl_file = osp.join(dataset_dir, 'cls/%s.mat' % did)
#                 self.files[split].append({
#                     'img': img_file,
#                     'lbl': lbl_file,
#                 })

#     def __getitem__(self, index):
#         data_file = self.files[self.split][index]
#         # load image
#         img_file = data_file['img']
#         img = PIL.Image.open(img_file)
#         img = np.array(img, dtype=np.uint8)
#         # load label
#         lbl_file = data_file['lbl']
#         mat = scipy.io.loadmat(lbl_file)
#         lbl = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)
#         lbl[lbl == 255] = -1
#         if self._transform:
#             return self.transform(img, lbl)
#         else:
#             return img, lbl
