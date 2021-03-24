'''MIT License
Copyright (C) 2020 Prokofiev Kirill, Intel Corporation
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.'''

import os
import re

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset


class CasiaSurfDataset(Dataset):
    PROTOCOLS = {'train': 'train', 'dev': 'dev_ref', 'test': 'test_res'}

    def __init__(self, protocol: int, dir_: str = 'data/CASIA_SURF', mode: str = 'train', depth=False, ir=False,
                 transform=None):
        self.dir = dir_
        self.mode = mode
        submode = PROTOCOLS[mode]
        file_name = '4@{}_{}.txt'.format(protocol, submode)
        with open(os.path.join(dir_, file_name), 'r') as file:
            self.items = []
            for line in file:
                if self.mode == 'train':
                    img_name, label = tuple(line[:-1].split(' '))
                    self.items.append(
                        (self.get_all_modalities(img_name, depth, ir), label))

                elif self.mode == 'dev':
                    folder_name, label = tuple(line[:-1].split(' '))
                    profile_dir = os.path.join(
                        self.dir, folder_name, 'profile')
                    for file in os.listdir(profile_dir):
                        img_name = os.path.join(folder_name, 'profile', file)
                        self.items.append(
                            (self.get_all_modalities(img_name, depth, ir), label))

                elif self.mode == 'test':
                    folder_name = line[:-1].split(' ')[0]
                    profile_dir = os.path.join(
                        self.dir, folder_name, 'profile')
                    for file in os.listdir(profile_dir):
                        img_name = os.path.join(folder_name, 'profile', file)
                        self.items.append(
                            (self.get_all_modalities(img_name, depth, ir), -1))

        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_names, label = self.items[idx]
        images = []
        for img_name in img_names:
            img_path = os.path.join(self.dir, img_name)
            img = cv.imread(img_path, flags=1)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            if self.transform is not None:
                img = self.transform(label=label, img=img)['image']
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)
            images += [torch.tensor(img)]

        return torch.cat(images, dim=0), 1-int(label)

    def get_all_modalities(self, img_path: str, depth: bool = True, ir: bool = True) -> list:
        result = [img_path]
        if depth:
            result += [re.sub('profile', 'depth', img_path)]
        if ir:
            result += [re.sub('profile', 'ir', img_path)]

        return result
