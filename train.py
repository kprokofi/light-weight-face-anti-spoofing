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
import os

import albumentations as A
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from eval_protocol import evaulate
from trainer import Trainer
from utils import (Transform, build_criterion, build_model, make_dataset,
                   make_loader, make_weights, read_py_config)


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='antispoofing training')
    parser.add_argument('--GPU', type=int, default=0, help='specify which gpu to use')
    parser.add_argument('--save_checkpoint', type=bool, default=True, 
                        help='whether or not to save your model')
    parser.add_argument('--config', type=str, default=None, required=True, 
                        help='Configuration file')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'], 
                        help='if you want to train model on cpu, pass "cpu" param')
    args = parser.parse_args()

    # manage device, arguments, reading config
    path_to_config = args.config
    config = read_py_config(path_to_config)
    device = args.device + f':{args.GPU}' if args.device == 'cuda' else 'cpu'
    if config.data_parallel.use_parallel:
        device = f'cuda:{config.data_parallel.parallel_params.output_device}'

    # launch training, validation, testing
    train(config, device, args.save_checkpoint)

def train(config, device='cuda:0', save_checkpoint=True):
    ''' procedure launching all main functions of training, validation and testing pipelines'''
    # preprocessing data
    normalize = A.Normalize(**config.img_norm_cfg)
    train_transform_real = A.Compose([
                            A.Resize(**config.resize, interpolation=cv.INTER_CUBIC),
                            A.HorizontalFlip(p=0.5),
                            A.augmentations.transforms.ISONoise(color_shift=(0.15,0.35), intensity=(0.2, 0.5), p=0.2),
                            A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, 
                                                                                brightness_by_max=True, always_apply=False, p=0.3),
                            A.augmentations.transforms.MotionBlur(blur_limit=5, p=0.2),
                            normalize
                            ])
    train_transform_spoof = A.Compose([
                            A.Resize(**config.resize, interpolation=cv.INTER_CUBIC),
                            A.HorizontalFlip(p=0.5),
                            A.augmentations.transforms.ISONoise(color_shift=(0.15,0.35), intensity=(0.2, 0.5), p=0.2),
                            A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, 
                                                                                brightness_by_max=True, always_apply=False, p=0.3),
                            A.augmentations.transforms.MotionBlur(blur_limit=5, p=0.2),
                            normalize
                            ])
    val_transform = A.Compose([
                A.Resize(**config.resize, interpolation=cv.INTER_CUBIC),
                normalize
                ])
    
    # load data
    sampler = config.data.sampler
    if sampler:
        num_instances, weights = make_weights(config)
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_instances, replacement=True)
    train_transform = Transform(train_spoof=train_transform_spoof, train_real=train_transform_real, val=None)
    val_transform = Transform(train_spoof=None, train_real=None, val=val_transform)
    train_dataset, val_dataset = make_dataset(config, train_transform, val_transform)
    train_loader, val_loader = make_loader(train_dataset, val_dataset, config, sampler=sampler)
    test_dataset = make_dataset(config, val_transform=val_transform, mode='eval')
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.data.batch_size,
                                                shuffle=True, pin_memory=config.data.pin_memory,
                                                num_workers=config.data.data_loader_workers)
    
    # build model and put it to cuda and if it needed then wrap model to data parallel
    model = build_model(config, device=device, strict=False, mode='train')
    model.to(device)
    if config.data_parallel.use_parallel:
        model = torch.nn.DataParallel(model, **config.data_parallel.parallel_params)
    
    # build a criterion
    SM = build_criterion(config, device, task='main').to(device)
    CE = build_criterion(config, device, task='rest').to(device)
    BCE = nn.BCELoss().to(device)
    criterion = (SM, CE, BCE) if config.multi_task_learning else SM
    
    # build optimizer and scheduler for it
    optimizer = torch.optim.SGD(model.parameters(), **config.optimizer)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config.scheduler)

    # create Trainer object and get experiment information
    trainer = Trainer(model, criterion, optimizer, device, config, train_loader, val_loader, test_loader)
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
        trainer.eval(epoch, accuracy, save_checkpoint=save_checkpoint)
        # for testing purposes 
        if config.test_steps:
            break

    # evaulate in the end of training    
    if config.evaulation:
        trainer.test(val_transform, file_name='LCC_FASD.txt', flag=None)
        trainer.test(val_transform, file_name='Celeba_test.txt', flag=True)

if __name__=='__main__':
    main()
