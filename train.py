import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
import MobilNet2
from amsoftmax import AMSoftmaxLoss, AngleSimpleLinear
from reader_dataset import LCFAD
from torch.autograd import Variable
import numpy as np
import cv2 as cv
import albumentations as A
from tqdm import tqdm
from label_smoothing import LabelSmoothingLoss, CrossEntropyReduction
import config
from utils import AverageMeter, read_py_config, save_checkpoint, precision
import os
from check_test import evaulate
from mobilenetv3 import mobilenetv3_large, h_swish

parser = argparse.ArgumentParser(description='antispoofing training')
current_dir = os.path.dirname(os.path.abspath(__file__))

# parse arguments
parser.add_argument('--GPU', type=int, default=1, help='specify which gpu to use')
parser.add_argument('--print-freq', '-p', default=20, type=int, help='print frequency (default: 20)')
parser.add_argument('--save_checkpoint', type=bool, default=True, help='whether or not to save your model')
parser.add_argument('--config', type=str, default=os.path.join(current_dir, 'config.py'), required=False,
                        help='Configuration file')

#global variables and argument parsing
args = parser.parse_args()
config = read_py_config(args.config)
experiment_snapshot = config['checkpoint']['snapshot_name']
experiment_path = config['checkpoint']['experiment_path']
WRITER = SummaryWriter(experiment_path)
STEP, VAL_STEP = 0, 0
BEST_ACCURACY, BEST_AUC, BEST_EER = 0, 0, 1000

def main():
    global args, BEST_ACCURACY, BEST_EER, BEST_AUC, config

    # loading data
    normalize = A.Normalize(**config['img_norm_cfg'])
    train_transform = A.Compose([
                            A.Resize(**config['resize']),
                            A.HorizontalFlip(p=0.5),
                            A.Rotate(limit=(-30, 30), p=0.5),
                            A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
                            normalize,
                            ])

    val_transform = A.Compose([
                A.Resize(**config['resize']),
                normalize,
                ])     

    train_dataset = LCFAD(root_dir=config['data']['data_root'], train=True, transform=train_transform)
    val_dataset = LCFAD(root_dir=config['data']['data_root'], train=False, transform=val_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['data']['batch_size'], 
                                                    shuffle=True, pin_memory=config['data']['pin_memory'], 
                                                    num_workers=config['data']['data_loader_workers'])

    val_loader = DataLoader(dataset=val_dataset, batch_size=config['data']['batch_size'], 
                                                shuffle=True, pin_memory=config['data']['pin_memory'], 
                                                num_workers=config['data']['data_loader_workers'])

    # model
    if config['model']['model_type'] == 'Mobilenet2':
        model = MobilNet2.MobileNetV2(use_amsoftmax=config['model']['use_amsoftmax'])
    else:
        assert config['model']['model_type'] == 'Mobilenet3'
        model = mobilenetv3_large()
        if config['model']['pretrained']:
            model.load_state_dict(torch.load('pretrained/mobilenetv3-large-1cd25616.pth', map_location=f'cuda:{args.GPU}'), strict=False)
            if config['loss']['loss_type'] == 'amsoftmax':
                model.classifier[3] = AngleSimpleLinear(1280, 2)
            else:
                model.classifier[3] = nn.Linear(1280, 2)

    #criterion
    if config['loss']['loss_type'] == 'amsoftmax':
        criterion = AMSoftmaxLoss(**config['amsoftmax'])
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.parameters(), **config['optimizer'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **config['schedular'])

    # learning epochs
    for epoch in range(config['epochs']['start_epoch'], config['epochs']['max_epoch']):
        if epoch != config['epochs']['start_epoch']:
            scheduler.step()

        # train for one epoch
        train_loss, train_accuracy = train(train_loader, model, criterion, optimizer, epoch)
        print(f'epoch: {epoch}  train loss: {train_loss}   train accuracy: {train_accuracy}')

        # evaluate on validation set
        accuracy = validate(val_loader, model, criterion)

        # remember best accuracy, AUC and EER and save checkpoint
        if accuracy > BEST_ACCURACY and args.save_checkpoint:
            AUC, EER, _, _, _ = evaulate(model, val_loader, compute_accuracy=False, GPU=args.GPU)
            if EER < BEST_EER or AUC > BEST_AUC:
                checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':epoch}
                save_checkpoint(checkpoint, f'{experiment_path}/{experiment_snapshot}')
                BEST_ACCURACY = max(accuracy, BEST_ACCURACY)
                print(f'epoch: {epoch}   AUC: {AUC}   EER: {EER}')
            BEST_EER = min(EER, BEST_EER)
            BEST_AUC = max(AUC, BEST_AUC)

        # evaluate on val every 30 epoch and save snapshot if better results achieved
        if (epoch%30 == 0 or epoch == config['epochs']['max_epoch']) and args.save_checkpoint:
            AUC, EER, _, _, _ = evaulate(model, val_loader, compute_accuracy=False, GPU=args.GPU)
            print(f'epoch: {epoch}   AUC: {AUC}   EER: {EER}')
            if EER < BEST_EER or AUC > BEST_AUC:
                checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':epoch}
                save_checkpoint(checkpoint, f'{experiment_path}/{experiment_snapshot}')
            BEST_EER = min(EER, BEST_EER)
            BEST_AUC = max(AUC, BEST_AUC)

        print(f'best val accuracy:  {BEST_ACCURACY}  best AUC: {BEST_AUC}  best EER: {BEST_EER}')
        
def train(train_loader, model, criterion, optimizer, epoch):
    global STEP, args, config
    # meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    # model to cuda, criterion to cuda
    model.cuda(device=args.GPU)
    criterion.cuda(device=args.GPU)

    # switch to train mode and train one epoch
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for i, (input, target) in loop:
        if config['data']['cuda']:
            input = input.cuda(device=args.GPU)
            target = target.cuda(device=args.GPU)

        # compute output and loss
        output = model(input)
        if config['loss']['loss_type'] == 'amsoftmax':
            new_target = F.one_hot(target, num_classes=2)
            loss = criterion(output, new_target)
        else:
            assert args.loss == 'cross_entropy'
            loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc = precision(output.data, target, s=config['amsoftmax']['s'])
        losses.update(loss.item(), input.size(0))
        accuracy.update(acc, input.size(0))

        # write to writer for tensorboard
        WRITER.add_scalar('Train/loss', losses.avg, global_step=STEP)
        WRITER.add_scalar('Train/accuracy',  accuracy.avg, global_step=STEP)
        STEP += 1

        # update progress bar
        max_epochs = config['epochs']['max_epoch']
        loop.set_description(f'Epoch [{epoch}/{max_epochs}]')
        if i % args.print_freq == 0:
            loop.set_postfix(loss=loss.item(), avr_loss = losses.avg, acc=acc, avr_acc=accuracy.avg, lr=optimizer.param_groups[0]['lr'])
    return losses.avg, accuracy.avg

def validate(val_loader, model, criterion):
    global args, VAL_STEP, config
    # meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    # switch to evaluation mode and inference the model
    model.eval()
    loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
    for i, (input, target) in loop:
        if config['data']['cuda']:
            input = input.cuda(device=args.GPU)
            target = target.cuda(device=args.GPU)

        # computing output and loss
        with torch.no_grad():
            output = model(input)
            if config['loss']['loss_type'] == 'amsoftmax':
                new_target = F.one_hot(target, num_classes=2)
                loss = criterion(output, new_target)
            else:
                assert config['loss']['loss_type'] == 'cross_entropy'
                loss = criterion(output, target)
        
        # measure accuracy and record loss
        acc = precision(output.data, target, s=config['amsoftmax']['s'])
        losses.update(loss.item(), input.size(0))
        accuracy.update(acc, input.size(0))

        # write val in writer
        WRITER.add_scalar('Val/loss', losses.avg, global_step=VAL_STEP)
        WRITER.add_scalar('Val/accuracy',  accuracy.avg, global_step=VAL_STEP)
        VAL_STEP += 1

        # update progress bar
        if i % args.print_freq == 0:
            loop.set_postfix(loss=loss.item(), avr_loss = losses.avg, acc=acc, avr_acc=accuracy.avg)

    print(f'val accuracy on epoch: {round(accuracy.avg, 3)}')

    return accuracy.avg

if __name__=='__main__':
    main()