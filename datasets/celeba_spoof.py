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

import json
import os

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset


class CelebASpoofDataset(Dataset):
    def __init__(self, root_folder, test_mode=False, transform=None, multi_learning=True):
        self.root_folder = root_folder
        if test_mode:
            list_path = os.path.join(root_folder, 'metas/intra_test/items_test.json')
        else:
            list_path = os.path.join(root_folder, 'metas/intra_test/items_train.json')

        with open(list_path, 'r') as f:
            self.data = json.load(f)
        # transform is supposed to be instance of the Transform object from utils.py pending entry label
        self.transform = transform
        self.multi_learning = multi_learning

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[str(idx)]
        img = cv.imread(os.path.join(self.root_folder, data_item['path']))
        bbox = data_item['bbox']

        real_h, real_w, _ = img.shape
        x1 = clamp(int(bbox[0]*(real_w / 224)), 0, real_w)
        y1 = clamp(int(bbox[1]*(real_h / 224)), 0, real_h)
        w1 = int(bbox[2]*(real_w / 224))
        h1 = int(bbox[3]*(real_h / 224))

        cropped_face = img[y1 : clamp(y1 + h1, 0, real_h), x1 : clamp(x1 + w1, 0, real_w), :]
        cropped_face = cv.cvtColor(cropped_face, cv.COLOR_BGR2RGB)

        if self.multi_learning:
            labels = data_item['labels']
            label = int(labels[43])
            real_labels = tuple(map(int, labels[0:40]))
            labels = [label, int(labels[40]), int(labels[41]), *real_labels]
        else:
            labels = int(data_item['labels'][43])
            label = labels

        if self.transform:
            cropped_face = self.transform(label=label, img=cropped_face)['image']
        cropped_face = np.transpose(cropped_face, (2, 0, 1)).astype(np.float32)
        return (torch.tensor(cropped_face), torch.tensor(labels, dtype=torch.long))

def clamp(x, min_x, max_x):
    return min(max(x, min_x), max_x)
