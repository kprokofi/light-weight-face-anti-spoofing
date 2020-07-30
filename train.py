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

# parse arguments
parser = argparse.ArgumentParser(description='antispoofing training')
parser.add_argument('-b', '--batch-size', default=100, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--cuda', type=bool, default=True, help='use cpu')
parser.add_argument('--GPU', type=int, default=0, help='specify which gpu to use')
parser.add_argument('--print-freq', '-p', default=20, type=int, help='print frequency (default: 20)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, help='initial learning rate')
parser.add_argument('--adjust_lr', type=list, default=[100, 150], help='spicify range of dropping lr')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--gamma', default=0.1, type=float, help='specify koefficient of lr dropping')
parser.add_argument('--save_checkpoint', type=bool, default=False, help='whether or not to save your model')
parser.add_argument('--loss', default='cross_entropy', type=str, help='which loss to use')
parser.add_argument('--classes', default=2, type=int, help='number of classes')

#global variables
WRITER = SummaryWriter(f'/home/prokofiev/pytorch/antispoofing/log_tensorboard/MobileNet_LCFAD_1.5')
STEP, VAL_STEP = 0, 0
BEST_ACCURACY = 0

def main():
    global args, BEST_ACCURACY

    # argements parsing
    args = parser.parse_args()
    
    # loading data
    normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = A.Compose([
                            # cv.cvtColor(code=cv.COLOR_BGR2RGB),
                            A.Resize(224, 224),
                            A.RandomCrop(224,224),
                            A.HorizontalFlip(p=0.5),
                            A.Rotate(limit=(-90, 90), p=0.5),
                            A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
                            normalize,
                            # A.pytorch.transforms.ToTensor()
                            ])

    val_transform = A.Compose([
                # cv.cvtColor(image, cv.COLOR_BGR2RGB),
                A.Resize(224, 224),
                normalize,
                # A.pytorch.transforms.ToTensor()
                ])     

    train_dataset = LCFAD(root_dir='/home/prokofiev/pytorch/LCC_FASD', train=True, transform=train_transform)
    val_dataset = LCFAD(root_dir='/home/prokofiev/pytorch/LCC_FASD', train=False, transform=val_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # model
    if args.loss == 'amsoftmax':
        model = MobilNet2.MobileNetV2(use_amsoftmax=True)
    else:
        assert args.loss == 'cross_entropy'
        model = MobilNet2.MobileNetV2()
    
    #criterion
    if args.loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = AMSoftmaxLoss(m=0.5, margin_type='cos', s=5)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.adjust_lr, gamma=args.gamma)

    # learning epochs
    for epoch in range(args.start_epoch, args.epochs):
        if epoch != args.start_epoch:
            scheduler.step()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        accuracy = validate(val_loader, model, criterion)

        # remember best accuracy and save checkpoint
        if accuracy > BEST_ACCURACY and args.save_checkpoint:
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint, 'my_best_modelMobileNet2_1.5.pth.tar')

        BEST_ACCURACY = max(accuracy, BEST_ACCURACY)
        print(f'best accuracy:  {BEST_ACCURACY}')
        

def train(train_loader, model, criterion, optimizer, epoch):
    global STEP, args
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
        if args.cuda:
            input = input.cuda(device=args.GPU)
            target = target.cuda(device=args.GPU)

        # compute output and loss
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc = precision(output.data, target)
        losses.update(loss.item(), input.size(0))
        accuracy.update(acc, input.size(0))

        # write to writer for tensorboard
        WRITER.add_scalar('Train/loss', losses.avg, global_step=STEP)
        WRITER.add_scalar('Train/accuracy',  accuracy.avg, global_step=STEP)
        STEP += 1

        # update progress bar
        loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
        if i % args.print_freq == 0:
            loop.set_postfix(loss=loss.item(), avr_loss = losses.avg, acc=acc, avr_acc=accuracy.avg)

def validate(val_loader, model, criterion):
    global args, VAL_STEP
    # meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    # switch to evaluation mode and inference the model
    loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
    for i, (input, target) in loop:
        if args.cuda:
            input = input.cuda(device=args.GPU)
            target = target.cuda(device=args.GPU)

        # computing output and loss
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)
        
        # measure accuracy and record loss
        acc = precision(output.data, target)
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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, filename="my_model.pth.tar"):
    print('==> saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(checkpoint, net, optimizer, load_optimizer=False):
    print("==> Loading checkpoint")
    net.load_state_dict(checkpoint['state_dict'])
    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

def precision(output, target):
    """Computes the precision"""
    accuracy = (output.argmax(dim=1) == target).float().mean().item()
    return accuracy*100

if __name__=='__main__':
    main()