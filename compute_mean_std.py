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

import argparse

import albumentations as A
import cv2 as cv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CelebASpoofDataset
from utils import Transform


def main():
    parser = argparse.ArgumentParser(description='mean and std computing')
    parser.add_argument('--root', type=str, default=None, required=True,
                        help='path to root folder of the CelebA_Spoof')
    parser.add_argument('--img_size', type=tuple, default=(128,128), required=False,
                        help='height and width of the image to resize')
    args = parser.parse_args()
    # transform image
    transforms = A.Compose([
                                A.Resize(*args.img_size, interpolation=cv.INTER_CUBIC),
                                A.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
                                ])
    root_folder = args.root
    train_dataset = CelebASpoofDataset(root_folder, test_mode=False,
                                       transform=Transform(transforms),
                                       multi_learning=False)
    dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    mean, std = compute_mean_std(dataloader)
    print(f'mean:{mean}, std:{std}')

def compute_mean_std(loader):
    ''' based on next formulas: E[x] = sum(x*p) = sum(x)/N, D[X] = E[(X-E(X))**2] = E[X**2] - (E[x])**2'''
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(loader, leave=False):
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5
    return mean, std

if __name__=="__main__":
    main()
