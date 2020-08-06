import os
import re

import cv2 as cv
import torch
from torch.utils.data import Dataset
import numpy as np


class CasiaSurfDataset(Dataset):
    def __init__(self, protocol: int, dir: str = 'data/CASIA_SURF', mode: str = 'train', depth=False, ir=False,
                 transform=None):
        self.dir = dir
        self.mode = mode
        submode = {'train': 'train', 'dev': 'dev_ref',
                   'test': 'test_res'}[mode]
        file_name = '4@{}_{}.txt'.format(protocol, submode)
        with open(os.path.join(dir, file_name), 'r') as file:
            lines = file.readlines()
            self.items = []
            for line in lines:
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
                img = self.transform(image=img)['image']
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)
            images += [torch.tensor(img)]

        return torch.cat(images, dim=0), int(label)

    def get_all_modalities(self, img_path: str, depth: bool = True, ir: bool = True) -> list:
        result = [img_path]
        if depth:
            result += [re.sub('profile', 'depth', img_path)]
        if ir:
            result += [re.sub('profile', 'ir', img_path)]

        return result