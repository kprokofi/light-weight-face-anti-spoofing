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
    # print experiments param
    init_experiment(config, path_to_config)
    # preprocessing data
    normalize = A.Normalize(**config['img_norm_cfg'])
    train_transform_real = A.Compose([
                            A.Resize(**config['resize'], interpolation=cv2.INTER_CUBIC),
                            A.HorizontalFlip(p=0.5),
                            A.augmentations.transforms.Blur(blur_limit=3, p=0.2),
                            A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.3),
                            A.augmentations.transforms.MotionBlur(blur_limit=4, p=0.2),
                            # A.augmentations.transforms.RGBShift(p=0.2),
                            A.augmentations.transforms.ISONoise(color_shift=(0.15,0.35), intensity=(0.2, 0.5)),
                            normalize,
                            ])

    train_transform_spoof = A.Compose([
                            A.Resize(**config['resize'], interpolation=cv2.INTER_CUBIC),
                            A.HorizontalFlip(p=0.5),
                            A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.3),
                            # A.augmentations.transforms.RGBShift(p=0.2),
                            A.augmentations.transforms.MotionBlur(blur_limit=4, p=0.2),
                            A.augmentations.transforms.ISONoise(color_shift=(0.15,0.35), intensity=(0.2, 0.5), p=0.2),
                            normalize,
                            ])

    val_transform = A.Compose([
                A.Resize(**config['resize'], interpolation=cv2.INTER_CUBIC),
                normalize,
                ])
    
    # load data
    sampler = config['data']['sampler']
    print(f'SAMLPER:{sampler}')
    if sampler:
        weights = make_weights(config)
        sampler = torch.utils.data.WeightedRandomSampler(weights, 494185, replacement=True)
    train_transform = Transform(train_spoof=train_transform_spoof, train_real=train_transform_real)
    val_transform = Transform(val=val_transform)
    train_dataset, val_dataset = make_dataset(config, train_transform, val_transform)
    train_loader, val_loader = make_loader(train_dataset, val_dataset, config, sampler=sampler)
    test_dataset = make_dataset(config, val_transform=val_transform, mode='eval')
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['data']['batch_size'],
                                                shuffle=True, pin_memory=config['data']['pin_memory'],
                                                num_workers=config['data']['data_loader_workers'])
    # build model and put it to cuda and if it needed then wrap model to data parallel
    model = build_model(config, args, strict=False)
    model.cuda(args.GPU)

    if config['data_parallel']['use_parallel']:
        model = torch.nn.DataParallel(model, **config['data_parallel']['parallel_params'])
    
    # build a criterion
    SM = build_criterion(config, args).cuda(device=args.GPU)
    CE = nn.CrossEntropyLoss().cuda(device=args.GPU)
    BCE = nn.BCELoss().cuda(device=args.GPU)
    criterion = (SM, CE, BCE)
    # build optimizer and scheduler for it
    optimizer = torch.optim.SGD(model.parameters(), **config['optimizer'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config['scheduler'])

    # learning epochs
    for epoch in range(config['epochs']['start_epoch'], config['epochs']['max_epoch']):
        if epoch != config['epochs']['start_epoch']:
            scheduler.step()

        # train for one epoch
        train_loss, train_accuracy = train(train_loader, model, criterion, optimizer, epoch)
        print(f'epoch: {epoch}  train loss: {train_loss}   train accuracy: {train_accuracy}')

        # evaluate on validation set
        accuracy = validate(val_loader, model, criterion)

        # remember best accuracy, AUC, EER, ACER and save checkpoint
        if (epoch == 0 or epoch >=60) and accuracy > BEST_ACCURACY and args.save_checkpoint:
            AUC, EER, _ , apcer, bpcer, acer = evaulate(model, val_loader, config, args, compute_accuracy=False)
            print(f'epoch: {epoch}   AUC: {AUC}   EER: {EER}   APCER: {apcer}   BPCER: {bpcer}   ACER: {acer}')
            if acer < BEST_ACER:
                BEST_ACER = acer
                checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':epoch}
                save_checkpoint(checkpoint, f'{experiment_path}/{experiment_snapshot}')
                BEST_ACCURACY = accuracy
                BEST_EER = EER
                BEST_AUC = AUC
            
        # evaluate on val every 10 epoch and save snapshot if better results achieved
        if ((epoch%10 == 0) or (epoch == config['epochs']['max_epoch']-1)) and args.save_checkpoint:
            # printing results
            AUC, EER, accur, apcer, bpcer, acer, _, _ = evaulate(model, test_loader, config, args, compute_accuracy=True) 
            print(f'epoch: {epoch}  accur: {round(np.mean(accur),3)}   AUC: {AUC}   EER: {EER}   APCER: {apcer}   BPCER: {bpcer}   ACER: {acer}')
    
    # evaulate in the end of training    
    if config['evaulation']:
        eval_model(model, config, val_transform, map_location = args.GPU, eval_func = evaulate, file_name='LCC_FASD.txt', flag=None)
        eval_model(model, config, val_transform, map_location = args.GPU, eval_func = evaulate, file_name='Celeba_test.txt', flag=True)

def train(train_loader, model, criterion, optimizer, epoch):
    global STEP, args, config
    # meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    # switch to train mode and train one epoch
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for i, (input, target) in loop:
        if config['data']['cuda']:
            input = input.cuda(device=args.GPU)
            target = target.cuda(device=args.GPU)
        # compute output and loss
        if config['aug']['type_aug'] == 'mixup':
            aug_output = mixup_target(input, target[:,i], config, args.GPU)

        if config['aug']['type_aug'] == 'cutmix':
            aug_output = cutmix(input, target[:,i], config, args)

        if config['loss']['loss_type'] == 'amsoftmax':
            if config['aug']['type_aug'] != None:
                input, targets = aug_output
                input, target_a, target_b, hot_target, lam = make_output(model, input, targets[:,0], config)
                new_target = (target_a, target_b, hot_target, lam)
                loss = multi_task_criterion(output, new_target, config, criterion, optimizer)
            else:
                new_target = F.one_hot(target[:,0], num_classes=2)
                output = make_output(model, input, new_target, config)
                loss = multi_task_criterion(output, target, config, criterion, optimizer)

        elif config['loss']['loss_type'] == 'cross_entropy':
            if config['aug']['type_aug'] != None:
                input, y_a, y_b, lam = aug_output
                new_target = F.one_hot(target[:,0], num_classes=2)
                output = make_output(model, input, new_target, config)
                loss = mixup_criterion(criterion, output, y_a, y_b, lam, config, optimizer)
            else:
                output = model.make_logits(model(input)) # must be tuple of outputs
                loss = multi_task_criterion(output, target, config, criterion, optimizer)
        else:
            assert config['loss']['loss_type'] == 'soft_triple'
            new_target = F.one_hot(target, num_classes=2)
            output = make_output(model, input, new_target, config)
            loss = multi_task_criterion(output, target, config, criterion, optimizer)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc = precision(output[0], target[:,0].reshape(-1), s=config['loss']['amsoftmax']['s'])
        losses.update(loss.item(), input.size(0))
        accuracy.update(acc, input.size(0))

        # write to writer for tensorboard
        WRITER.add_scalar('Train/loss', loss, global_step=STEP)
        WRITER.add_scalar('Train/accuracy',  accuracy.avg, global_step=STEP)
        STEP += 1

        # update progress bar
        max_epochs = config['epochs']['max_epoch']
        loop.set_description(f'Epoch [{epoch}/{max_epochs}]')
        loop.set_postfix(loss=loss.item(), avr_loss = losses.avg, acc=acc, avr_acc=accuracy.avg, lr=optimizer.param_groups[0]['lr'])
    return losses.avg, accuracy.avg

def validate(val_loader, model, criterion):
    global args, VAL_STEP, config
    # meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    # multiple loss
    if type(criterion) == tuple:
        criterion = criterion[0]  
    # switch to evaluation mode and inference the model
    model.eval()
    loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
    for i, (input, target) in loop:
        if config['data']['cuda']:
            input = input.cuda(device=args.GPU)
            if len(target.shape) > 1:
                target = target[:, 0].reshape(-1).cuda(device=args.GPU)

        # computing output and loss
        with torch.no_grad():
            features = model(input)
            if config['data_parallel']['use_parallel']:
                model1 = model.module
            else:
                model1 = model
            
            output = model1.make_logits(features)
            if type(output) == tuple:
                output = output[0]
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
        loop.set_postfix(loss=loss.item(), avr_loss = losses.avg, acc=acc, avr_acc=accuracy.avg)

    print(f'val accuracy on epoch: {round(accuracy.avg, 3)}, loss on epoch:{round(losses.avg, 3)}')
    # write val in writer
    WRITER.add_scalar('Val/loss', losses.avg, global_step=VAL_STEP)
    WRITER.add_scalar('Val/accuracy',  accuracy.avg, global_step=VAL_STEP)
    VAL_STEP += 1

    return accuracy.avg

def mixup_criterion(criterion, pred, y_a, y_b, lam, config, optimizer):
    if type(pred) == tuple:
        pred = pred[0]
    return lam * criterion(pred, y_a, config, criterion, optimizer) + (1 - lam) * criterion(pred, y_a, config, criterion, optimizer)

def eval_model(model, config, transform, eval_func, file_name, map_location = 0, flag=None, save=True):
    if flag:
        config['test_dataset']['type'] = 'celeba-spoof'

    print('_____________EVAULATION_____________')
    # load snapshot
    path_to_experiment = os.path.join(config['checkpoint']['experiment_path'], config['checkpoint']['snapshot_name'])
    checkpoint = torch.load(path_to_experiment, map_location=torch.device(f'cuda:{map_location}')) 
    load_checkpoint(checkpoint, model, optimizer=None)
    epoch_of_checkpoint = checkpoint['epoch']
    # making dataset
    test_dataset = make_dataset(config, val_transform=transform, mode='eval')
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['data']['batch_size'],
                                                shuffle=True, pin_memory=config['data']['pin_memory'],
                                                num_workers=config['data']['data_loader_workers'])
    # printing results
    AUC, EER, accur, apcer, bpcer, acer, _, _ = evaulate(model, test_loader, config, args, compute_accuracy=True)
    results = f'''accuracy on test data = {round(np.mean(accur),3)}     AUC = {round(AUC,3)}     EER = {round(EER*100,2)}     apcer = {round(apcer*100,2)}     bpcer = {round(bpcer*100,2)}     acer = {round(acer*100,2)}   checkpoint made on {epoch_of_checkpoint} epoch'''    
   
    with open(os.path.join(config['checkpoint']['experiment_path'], file_name), 'w') as f:
        f.write(results)

def multi_task_criterion(output: tuple, target: torch.tensor, config, criterion, optimizer, C: float=1, Cs: float=0.1, Ci: float=0.01, Cf: float=1.):
    ''' output -> tuple of given losses
    target -> torch tensor of a shape [batch*num_tasks]
    return -> loss function '''
    SM, CE, BCE = criterion
    if config['aug']['type_aug']:
        # filter output and tsrget for
        if config['loss']['loss_type'] == 'amsoftmax':
            target_a, target_b, one_hot, lam = target
            spoof_loss = SM(output[0], one_hot)
        else:
            target_a, target_b, lam = target
            spoof_loss = mixup_criterion(SM, output[0], y_a=target_a[:,0].reshape(-1), 
                                                             y_b=target_b[:,0].reshape(-1),
                                                             lam=lam, config=config)

        spoof_type_loss = mixup_criterion(CE, output[1], y_a=target_a[:,1].reshape(-1), 
                                                             y_b=target_b[:,1].reshape(-1),
                                                             lam=lam, config=config)
        lightning_loss = mixup_criterion(CE, output[2], y_a=target_a[:,2].reshape(-1), 
                                                             y_b=target_b[:,2].reshape(-1),
                                                             lam=lam, config=config)
        real_atr_loss = mixup_criterion(BCE, output[3], y_a=target_a[:,3:].reshape(-1), 
                                                             y_b=target_b[:,3:].reshape(-1),
                                                             lam=lam, config=config)
    else:
        # spoof loss, take derivitive
        if config['loss']['loss_type'] == 'amsoftmax':
            spoof_target = F.one_hot(target[:,0], num_classes=2)
        else:
            spoof_target = target
        
        # compute losses
        spoof_loss = SM(output[0], spoof_target)
        spoof_type_loss =  CE(output[1], target[:,1].reshape(-1))
        lightning_loss =  CE(output[2], target[:,2].reshape(-1))
        # filter output for real images and compute third loss
        mask = target[:,0] == 0
        filtered_output = output[3][mask] 
        filtered_target = target[:,3:][mask].type(torch.float32)

        real_atr_loss = BCE(filtered_output, filtered_target)
        
    print(spoof_loss, spoof_type_loss, lightning_loss, real_atr_loss)
    # taking derivitives
    optimizer.zero_grad()
    spoof_loss.backward(retain_graph=True)

    optimizer.zero_grad()
    spoof_type_loss.backward(retain_graph=True)

    optimizer.zero_grad()
    lightning_loss.backward(retain_graph=True)

    optimizer.zero_grad()
    real_atr_loss.backward(retain_graph=True)
    # combine losses
    loss = C*spoof_loss + Cs*spoof_type_loss + Ci*lightning_loss + Cf*real_atr_loss
    exit()
    return loss

def init_experiment(config, path_to_config):
    print(f'_______INIT EXPERIMENT {os.path.splitext(path_to_config)[0][-2:]}______')
    print('\n\nSNAPSHOT')
    for key, item in config['checkpoint'].items():
            print(f'{key} --> {item}')
    print('\n\nMODEL')
    for key, item in config['model'].items():
            print(f'{key} --> {item}')
    loss_type = config['loss']['loss_type']
    print(f'\n\nLOSS TYPE : {loss_type.upper()}')
    if loss_type in ('amsoftmax', 'soft_triple'):
        for key, item in config['loss'][f'{loss_type}'].items():
            print(f'{key} --> {item}')
    print('\n\nDROPOUT PARAMS')
    for key, item in config['dropout'].items():
            print(f'{key} --> {item}')
    print('\n\nOPTIMAIZER')
    for key, item in config['optimizer'].items():
            print(f'{key} --> {item}')
    print('\n\nADDITIONAL USING PARAMETRS')
    if config['aug']['type_aug']:
        type_aug = config['aug']['type_aug']
        print(f'\nAUG TYPE = {type_aug} USING')
        for key, item in config['optimizer'].items():
            print(f'{key} --> {item}')
    if config['RSC']['use_rsc']:
        print(f'RSC USING')
        for key, item in config['RSC'].items():
            print(f'{key} --> {item}') 
    if config['data_parallel']['use_parallel']:
        ids = config['data_parallel']['parallel_params']['device_ids']
        print(f'USING DATA PATALLEL ON {ids[0]} and {ids[1]} GPU')
    if config['data']['sampler']:
        print('USING SAMPLER')
    if config['loss']['amsoftmax']['ratio'] != [1,1]:
        print('USING ADAPTIVE LOSS')

if __name__=='__main__':
    main()
