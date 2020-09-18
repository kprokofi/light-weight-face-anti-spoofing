'''MIT License

Copyright (C) 2020 Prokofiev Kirill
 
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import albumentations as A
from tqdm import tqdm
from utils import read_py_config, make_weights, Transform, make_dataset, make_loader, build_model, build_criterion
from trainer import Trainer
import os
from eval_protocol import evaulate
import cv2 as cv

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='antispoofing training')
    parser.add_argument('--GPU', type=int, default=1, help='specify which gpu to use')
    parser.add_argument('--print-freq', '-p', default=5, type=int, help='print frequency (default: 20)')
    parser.add_argument('--save_checkpoint', type=bool, default=True, help='whether or not to save your model')
    parser.add_argument('--config', type=str, default=None, required=True, help='Configuration file')
    args = parser.parse_args()

    # reade config
    path_to_config = args.config
    config = read_py_config(path_to_config)

    # preprocessing data
    normalize = A.Normalize(**config.img_norm_cfg)
    train_transform_real = A.Compose([
                            A.Resize(**config.resize, interpolation=cv.INTER_CUBIC),
                            A.HorizontalFlip(p=0.5),
                            A.augmentations.transforms.ISONoise(color_shift=(0.15,0.35), intensity=(0.2, 0.5), p=0.2),
                            A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.3),
                            A.augmentations.transforms.MotionBlur(blur_limit=5, p=0.2),
                            normalize,
                            ])

    train_transform_spoof = A.Compose([
                            A.Resize(**config.resize, interpolation=cv.INTER_CUBIC),
                            A.HorizontalFlip(p=0.5),
                            A.augmentations.transforms.ISONoise(color_shift=(0.15,0.35), intensity=(0.2, 0.5), p=0.2),
                            A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.3),
                            A.augmentations.transforms.MotionBlur(blur_limit=5, p=0.2),
                            normalize,
                            ])

    val_transform = A.Compose([
                A.Resize(**config.resize, interpolation=cv.INTER_CUBIC),
                normalize
                ])
    
    # load data
    sampler = config.data.sampler
    if sampler:
        weights = make_weights(config)
        sampler = torch.utils.data.WeightedRandomSampler(weights, 494185, replacement=True)
    train_transform = Transform(train_spoof=train_transform_spoof, train_real=train_transform_real, val=None)
    val_transform = Transform(train_spoof=None, train_real=None, val=val_transform)
    train_dataset, val_dataset = make_dataset(config, train_transform, val_transform)
    train_loader, val_loader = make_loader(train_dataset, val_dataset, config, sampler=sampler)
    test_dataset = make_dataset(config, val_transform=val_transform, mode='eval')
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.data.batch_size,
                                                shuffle=True, pin_memory=config.data.pin_memory,
                                                num_workers=config.data.data_loader_workers)

    # build model and put it to cuda and if it needed then wrap model to data parallel
    model = build_model(config, args, strict=False)
    model.cuda(args.GPU)
    if config.data_parallel.use_parallel:
        model = torch.nn.DataParallel(model, **config.data_parallel.parallel_params)
    
    # build a criterion
    SM = build_criterion(config, args, task='main').cuda(device=args.GPU)
    CE = build_criterion(config, args, task='rest').cuda(device=args.GPU)
    BCE = nn.BCELoss().cuda(device=args.GPU)
    criterion = (SM, CE, BCE) if config.multi_task_learning else SM
    
    # build optimizer and scheduler for it
    optimizer = torch.optim.SGD(model.parameters(), **config.optimizer)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config.scheduler)

    # create Trainer object and get experiment information
    trainer = Trainer(model, criterion, optimizer, args, config, train_loader, val_loader, test_loader)
    trainer.get_exp_info()

    # learning epochs
    for epoch in range(config.epochs.start_epoch, config.epochs.max_epoch):
        if epoch != config.epochs.start_epoch:
            scheduler.step()

        # train model for one epoch
        train_loss, train_accuracy = trainer.train(epoch)
        print(f'epoch: {epoch}  train loss: {train_loss}   train accuracy: {train_accuracy}')

        # validate your model
        accuracy = trainer.validate()

        # eval metrics such as AUC, APCER, BPCER, ACER on val and test dataset according to rule
        trainer.eval(epoch, accuracy) 

    # evaulate in the end of training    
    if config.evaulation:
        trainer.test(val_transform, file_name='LCC_FASD.txt', flag=None)
        trainer.test(val_transform, file_name='Celeba_test.txt', flag=True)

if __name__=='__main__':
    main()
