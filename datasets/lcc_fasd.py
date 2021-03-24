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

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset


class LccFasdDataset(Dataset):
    def __init__(self, root_dir, protocol='train', transform=None, get_img_path=False):
        assert protocol in ['train', 'val', 'test', 'combine_partly', 'val_test', 'combine_all']
        self.root_dir = root_dir
        self.transform = transform
        self.get_img_path = get_img_path
        if protocol == 'train':
            spoof_img, real_img = self.get_train_img(self.root_dir)
            self.list_img = spoof_img + real_img
            self.labels = torch.cat((torch.ones(len(spoof_img), dtype=torch.long),
                                    (torch.zeros(len(real_img), dtype=torch.long))))
        elif protocol == 'val':
            spoof_img, real_img = self.get_val_img(self.root_dir)
            self.list_img = spoof_img + real_img
            self.labels = torch.cat((torch.ones(len(spoof_img), dtype=torch.long),
                                    (torch.zeros(len(real_img), dtype=torch.long))))
        elif protocol == 'test':
            spoof_img, real_img = self.get_test_img(self.root_dir)
            self.list_img = spoof_img + real_img
            self.labels = torch.cat((torch.ones(len(spoof_img), dtype=torch.long),
                                        (torch.zeros(len(real_img), dtype=torch.long))))
        elif protocol == 'combine_partly':
            spoof_img_train, real_img_train = self.get_train_img(self.root_dir)
            spoof_img_val, real_img_val = self.get_val_img(self.root_dir)
            self.list_img = spoof_img_train + spoof_img_val + real_img_train + real_img_val
            self.labels = torch.cat((torch.ones(len(spoof_img_train)
                                    + len(spoof_img_val), dtype=torch.long),
                                    (torch.zeros(len(real_img_train)
                                    + len(real_img_val), dtype=torch.long))))
        elif protocol == 'val_test':
            spoof_img_test, real_img_test = self.get_test_img(self.root_dir)
            spoof_img_val, real_img_val = self.get_val_img(self.root_dir)
            self.list_img = spoof_img_test + spoof_img_val + real_img_test + real_img_val
            self.labels = torch.cat((torch.ones(len(spoof_img_test)
                                                + len(spoof_img_val), dtype=torch.long),
                                                (torch.zeros(len(real_img_test)
                                                + len(real_img_val), dtype=torch.long))))
        else:
            spoof_img_train, real_img_train = self.get_train_img(self.root_dir)
            spoof_img_val, real_img_val =self. get_val_img(self.root_dir)
            spoof_img_test, real_img_test = self.get_test_img(self.root_dir)
            self.list_img = (spoof_img_train + spoof_img_val + spoof_img_test
                                + real_img_train + real_img_val + real_img_test)
            self.labels = torch.cat((torch.ones(len(spoof_img_train) + len(spoof_img_val)
                                                + len(spoof_img_test), dtype=torch.long),
                                                (torch.zeros(len(real_img_train) + len(real_img_val)
                                                + len(real_img_test), dtype=torch.long))))
    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.list_img[index])
        image = cv.imread(img_path, flags=1)
        if self.get_img_path:
            return image, img_path
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        y_label = self.labels[index]
        if self.transform:
            image = self.transform(label=y_label, img=image)['image']
        # [batch, channels, height, width]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return torch.tensor(image), y_label

    @staticmethod
    def get_val_img(root_dir):
        name_of_real_img = filter(lambda x: x.endswith('.png') or x.endswith('.jpg'),
                        os.listdir(os.path.join(root_dir, 'LCC_FASD_development/real')))
        real_img = list(map(lambda x: os.path.join('LCC_FASD_development/real', x), name_of_real_img))
        name_of_spoof_img = filter(lambda x: x.endswith('.png') or x.endswith('.jpg'),
                        os.listdir(os.path.join(root_dir, 'LCC_FASD_development/spoof')))
        spoof_img = list(map(lambda x: os.path.join('LCC_FASD_development/spoof', x), name_of_spoof_img))
        return spoof_img, real_img

    @staticmethod
    def get_train_img(root_dir):
        name_of_real_img = filter(lambda x: x.endswith('.png') or x.endswith('.jpg'),
                        os.listdir(os.path.join(root_dir, 'LCC_FASD_training/real')))
        real_img = list(map(lambda x: os.path.join('LCC_FASD_training/real', x), name_of_real_img))

        name_of_spoof_img = filter(lambda x: x.endswith('.png') or x.endswith('.jpg'),
                        os.listdir(os.path.join(root_dir, 'LCC_FASD_training/spoof')))
        spoof_img = list(map(lambda x: os.path.join('LCC_FASD_training/spoof', x), name_of_spoof_img))
        return spoof_img, real_img

    @staticmethod
    def get_test_img(root_dir):
        name_of_real_img = filter(lambda x: x.startswith('real'),
                        os.listdir(os.path.join(root_dir, 'LCC_FASD_evaluation')))
        real_img = list(map(lambda x: os.path.join('LCC_FASD_evaluation', x), name_of_real_img))

        name_of_spoof_img = filter(lambda x: x.startswith('spoof'),
                            os.listdir(os.path.join(root_dir, 'LCC_FASD_evaluation')))
        spoof_img = list(map(lambda x: os.path.join('LCC_FASD_evaluation', x), name_of_spoof_img))
        return spoof_img, real_img
