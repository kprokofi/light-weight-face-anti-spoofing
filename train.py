import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from losses import AMSoftmaxLoss, AngleSimpleLinear, SoftTripleLoss, SoftTripleLinear
import albumentations as A
from tqdm import tqdm
from utils import *
import os
from eval_protocol import evaulate
import cv2

parser = argparse.ArgumentParser(description='antispoofing training')
current_dir = os.path.dirname(os.path.abspath(__file__))

# parse arguments
parser.add_argument('--GPU', type=int, default=1, help='specify which gpu to use')
parser.add_argument('--print-freq', '-p', default=5, type=int, help='print frequency (default: 20)')
parser.add_argument('--save_checkpoint', type=bool, default=True, help='whether or not to save your model')
parser.add_argument('--config', type=str, default='config.py', required=True,
                        help='Configuration file')

# global variables and argument parsing
args = parser.parse_args()
path_to_config = os.path.join(current_dir, args.config)
config = read_py_config(path_to_config)
experiment_snapshot = config['checkpoint']['snapshot_name']
experiment_path = config['checkpoint']['experiment_path']
WRITER = SummaryWriter(experiment_path)
STEP, VAL_STEP = 0, 0
BEST_ACCURACY, BEST_AUC, BEST_EER, BEST_ACER = 0, 0, float('inf'), float('inf')
def main():
    global args, BEST_ACCURACY, BEST_EER, BEST_AUC, BEST_ACER, config

    # preprocessing data
    normalize = A.Normalize(**config['img_norm_cfg'])
    train_transform = A.Compose([
                            A.Resize(**config['resize'], interpolation=cv2.INTER_CUBIC),
                            A.HorizontalFlip(p=0.5),
                            A.Rotate(limit=(-30, 30), p=0.5),
                            A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
                            normalize,
                            ])

    val_transform = A.Compose([
                A.Resize(**config['resize']),
                normalize,
                ])
    
    # load data
    train_dataset, val_dataset = make_dataset(config, train_transform, val_transform)
    train_loader, val_loader = make_loader(train_dataset, val_dataset, config)

    # build model
    model = build_model(config, args, strict=True)

    #criterion
    if config['loss']['loss_type'] == 'amsoftmax':
        criterion = AMSoftmaxLoss(**config['loss']['amsoftmax'])
    elif config['loss']['loss_type'] == 'soft_triple':
        criterion = SoftTripleLoss(**config['loss']['soft_triple'])
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
            AUC, EER, _ , apcer, bpcer, acer = evaulate(model, val_loader, config, args, compute_accuracy=False)
            print(f'epoch: {epoch}   AUC: {AUC}   EER: {EER}   APCER: {apcer}   BPCER: {bpcer}   ACER: {acer}')
            if EER < BEST_EER or acer < BEST_ACER:
                checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':epoch}
                save_checkpoint(checkpoint, f'{experiment_path}/{experiment_snapshot}')
                BEST_ACCURACY = max(accuracy, BEST_ACCURACY)
                BEST_EER = min(EER, BEST_EER)
                BEST_ACER = min(acer, BEST_ACER)
                BEST_AUC = max(AUC, BEST_AUC)
                print(f'epoch: {epoch}   AUC: {AUC}   EER: {EER}   APCER: {apcer}   BPCER: {bpcer}   ACER: {acer}')
            
        # evaluate on val every 10 epoch and save snapshot if better results achieved
        if (epoch%10 == 0 or epoch == config['epochs']['max_epoch']) and args.save_checkpoint:
            AUC, EER, _ , apcer, bpcer, acer = evaulate(model, val_loader, config, args, compute_accuracy=False)
            print(f'epoch: {epoch}   AUC: {AUC}   EER: {EER}   APCER: {apcer}   BPCER: {bpcer}   ACER: {acer}')
            if EER < BEST_EER or acer < BEST_ACER:
                checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':epoch}
                save_checkpoint(checkpoint, f'{experiment_path}/{experiment_snapshot}')
                BEST_EER = min(EER, BEST_EER)
                BEST_ACER = min(acer, BEST_ACER)
                BEST_AUC = max(AUC, BEST_AUC)
        print(f'best val accuracy:  {BEST_ACCURACY}  best AUC: {BEST_AUC}  best EER: {BEST_EER} best ACER: {BEST_ACER}')

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
        if config['aug']['type_aug'] == 'mixup':
            aug_output = mixup_target(input, target, config['aug']['alpha'], args.GPU, criterion=config['loss']['loss_type'])
        if config['aug']['type_aug'] == 'cutmix':
            aug_output = cutmix(input, target, config, args)
        if config['loss']['loss_type'] == 'amsoftmax':
            if config['aug']['type_aug'] != None:
                input, targets = aug_output
                output = model(input)
                loss = criterion(output, targets)
            else:
                output = model(input)
                new_target = F.one_hot(target, num_classes=2)
                loss = criterion(output, new_target)
        elif config['loss']['loss_type'] == 'cross_entropy':
            if config['aug']['type_aug'] != None:
                input, y_a, y_b, lam = aug_output
                output = model(input)
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            else:
                output = model(input)
                loss = criterion(output, target)
        else:
            assert config['loss']['loss_type'] == 'soft_triple'
            output = model(input)
            loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc = precision(output, target, s=config['loss']['amsoftmax']['s'])
        losses.update(loss.item(), input.size(0))
        accuracy.update(acc, input.size(0))

        # write to writer for tensorboard
        if i % args.print_freq == 0:
            WRITER.add_scalar('Train/loss', loss, global_step=STEP)
            WRITER.add_scalar('Train/accuracy',  acc, global_step=STEP)
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
                assert config['loss']['loss_type'] in ('cross_entropy', 'soft_triple')
                loss = criterion(output, target)

        # measure accuracy and record loss
        acc = precision(output, target, s=config['loss']['amsoftmax']['s'])
        losses.update(loss.item(), input.size(0))
        accuracy.update(acc, input.size(0))

        # update progress bar
        if i % args.print_freq == 0:
            loop.set_postfix(loss=loss.item(), avr_loss = losses.avg, acc=acc, avr_acc=accuracy.avg)

    print(f'val accuracy on epoch: {round(accuracy.avg, 3)}, loss on epoch:{round(losses.avg, 3)}')
    # write val in writer
    WRITER.add_scalar('Val/loss', losses.avg, global_step=VAL_STEP)
    WRITER.add_scalar('Val/accuracy',  accuracy.avg, global_step=VAL_STEP)
    VAL_STEP += 1

    return accuracy.avg

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    if type(pred) == tuple:
        pred = pred[0]
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

if __name__=='__main__':
    main()
