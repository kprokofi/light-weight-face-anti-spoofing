import os.path as osp
import sys
import torch
from importlib import import_module
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from datasets import LCFAD, CelebASpoofDataset, CasiaSurfDataset
from torch.utils.data import DataLoader
from losses import AngleSimpleLinear, SoftTripleLinear
import torch.nn as nn
from models import mobilenetv2, mobilenetv3_large, mobilenetv3_small

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

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))

def read_py_config(filename):
    filename = osp.abspath(osp.expanduser(filename))
    check_file_exist(filename)
    assert filename.endswith('.py')
    module_name = osp.basename(filename)[:-3]
    if '.' in module_name:
        raise ValueError('Dots are not allowed in config file path.')
    config_dir = osp.dirname(filename)
    sys.path.insert(0, config_dir)
    mod = import_module(module_name)
    sys.path.pop(0)
    cfg_dict = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
    }

    return cfg_dict

def save_checkpoint(state, filename="my_model.pth.tar"):
    print('==> saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(checkpoint, net, optimizer, load_optimizer=False):
    print("==> Loading checkpoint")
    net.load_state_dict(checkpoint['state_dict'])
    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

def precision(output, target, s=None):
    """Computes the precision"""
    if s:
        output = output*s
    if type(output) == tuple:
        output = output[0].data
    accuracy = (output.argmax(dim=1) == target).float().mean().item()
    return accuracy*100

def mixup_target(input, target, alpha, cuda, criterion='amsoftmax', num_classes=2):
 # compute mix-up augmentation
    input, target_a, target_b, lam = mixup_data(input, target, alpha, cuda)
    input, target_a, target_b = map(Variable, (input, target_a, target_b))
    # compute new target
    target_a_hot = F.one_hot(target_a, num_classes)
    target_b_hot = F.one_hot(target_b, num_classes)
    new_target = lam*target_a_hot + (1-lam)*target_b_hot
    if criterion == 'amsoftmax':
        return input, new_target
    else:
        assert criterion == 'cross_entropy'
        return input, target_a, target_b, lam

def mixup_data(x, y, alpha=1.0, cuda=0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda(device=cuda)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
    
def freeze_layers(model, open_layers):

    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

def make_dataset(config: dict, train_transform: object = None, val_transform: object = None, mode='train'):
    ''' make train and val datasets ''' 
    if config['dataset'] == 'LCCFASD':
        train =  LCFAD(root_dir=config['data']['data_root'], train=True, transform=train_transform)
        val = LCFAD(root_dir=config['data']['data_root'], train=False, transform=val_transform)
        test = val
    elif config['dataset'] == 'celeba-spoof':
        train =  CelebASpoofDataset(root_folder=config['data']['data_root'], test_mode=False, transform=train_transform)
        val = CelebASpoofDataset(root_folder=config['data']['data_root'], test_mode=True, transform=val_transform)
        test = val
    else:
        assert config['dataset'] == 'Casia'
        train = CasiaSurfDataset(protocol=1, dir=config['data']['data_root'], mode='train', transform=train_transform)
        val = CasiaSurfDataset(protocol=1, dir=config['data']['data_root'], mode='dev', transform=val_transform)
        test = CasiaSurfDataset(protocol=1, dir=config['data']['data_root'], mode='test', transform=val_transform)
    if mode == 'eval':
        return test
    return train, val

def make_loader(train, val, config):
    ''' make data loader from given train and val dataset
    train, val -> train loader, val loader'''

    train_loader = DataLoader(dataset=train, batch_size=config['data']['batch_size'],
                                                    shuffle=True, pin_memory=config['data']['pin_memory'],
                                                    num_workers=config['data']['data_loader_workers'])

    val_loader = DataLoader(dataset=val, batch_size=config['data']['batch_size'],
                                                shuffle=True, pin_memory=config['data']['pin_memory'],
                                                num_workers=config['data']['data_loader_workers'])
    return train_loader, val_loader

def build_model(config, args, strict=True):
    ''' build model and change layers depends on loss type'''

    if config['model']['model_type'] == 'Mobilenet2':
        model = mobilenetv2()
        if config['model']['pretrained']:
            model.load_state_dict(torch.load('pretrained/mobilenetv2_128x128-fd66a69d.pth', 
                                                map_location=f'cuda:{args.GPU}'), strict=strict)
        
        if config['loss']['loss_type'] == 'amsoftmax':
            model.conv = nn.Sequential(
                        nn.Conv2d(320, config['model']['embeding_dim'], 1, 1, 0, bias=False),
                        nn.Dropout(0.5),
                        nn.BatchNorm2d(config['model']['embeding_dim']),
                        nn.PReLU()
                    )
            model.classifier = AngleSimpleLinear(config['model']['embeding_dim'], 2)


        elif config['loss']['loss_type'] == 'soft_triple':
            model.conv = nn.Sequential(
                        nn.Conv2d(320, config['model']['embeding_dim'], 1, 1, 0, bias=False),
                        nn.Dropout(0.5),
                        nn.BatchNorm2d(config['model']['embeding_dim']),
                        nn.PReLU()
                    )
            model.classifier = SoftTripleLinear(config['model']['embeding_dim'], 2, num_proxies=config['loss']['soft_triple']['K'])
    else:
        assert config['model']['model_type'] == 'Mobilenet3'
        if config['model']['model_size'] == 'large':
            model = mobilenetv3_large()
            if config['model']['pretrained']:
                model.load_state_dict(torch.load('pretrained/mobilenetv3-large-1cd25616.pth', 
                                                map_location=f'cuda:{args.GPU}'), strict=strict)
        else:
            assert config['model']['model_size'] == 'small'
            model = mobilenetv3_small( width_mult=.75)
            if config['model']['pretrained']:
                model.load_state_dict(torch.load('pretrained/mobilenetv3-small-0.75-86c972c3.pth', 
                                                map_location=f'cuda:{args.GPU}'), strict=strict)

        if config['model']['model_size'] == 'small':
            exp_size = 432
        else:
            assert config['model']['model_size'] == 'large'
            exp_size = 960

        if config['loss']['loss_type'] == 'amsoftmax':
            model.classifier[0] = nn.Linear(exp_size, config['model']['embeding_dim'])
            model.classifier[2] = nn.BatchNorm1d(config['model']['embeding_dim'])
            model.classifier[4] = AngleSimpleLinear(config['model']['embeding_dim'], 2)
            
        elif config['loss']['loss_type'] == 'cross_entropy':
            model.classifier[1] == nn.Dropout(p=0.5)
            model.classifier[4] = nn.Linear(1280, 2)

        else:
            assert config['loss']['loss_type'] == 'soft_triple'
            model.classifier[0] = nn.Linear(exp_size, config['model']['embeding_dim'])
            model.classifier[2] = nn.BatchNorm1d(config['model']['embeding_dim'])
            model.classifier[4] = SoftTripleLinear(config['model']['embeding_dim'], 2, num_proxies=config['loss']['soft_triple']['K'])
    return model

def cutmix(input, target, config, args):
    r = np.random.rand(1)
    if (config['aug']['beta'] > 0) and (config['aug']['alpha'] > 0) and (r < config['aug']['cutmix_prob']):
        # generate mixed sample
        lam = np.random.beta(config['aug']['alpha'] > 0, config['aug']['beta'] > 0)
        rand_index = torch.randperm(input.size()[0]).cuda(device=args.GPU)
        # get one hot target vectors 
        target_a2 = F.one_hot(target) 
        target_b2 = F.one_hot(target)[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        # do merging classes (cutmix)
        new_target = lam*target_a2 + (1.0 - lam)*target_b2
        return input, new_target
    target = F.one_hot(target, num_classes=2)
    return input, target

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2