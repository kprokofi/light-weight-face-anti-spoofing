import os.path as osp
import sys
import torch
from importlib import import_module
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from models import mobilenetv2, mobilenetv3_large, mobilenetv3_small
import json
from losses import AngleSimpleLinear, SoftTripleLinear

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

def build_model(config, args, strict=True):
    ''' build model and change layers depends on loss type'''

    if config['model']['model_type'] == 'Mobilenet2':
        model = mobilenetv2(prob_dropout=config['dropout']['prob_dropout'])
        if config['model']['pretrained']:
            model.load_state_dict(torch.load('pretrained/mobilenetv2_128x128-fd66a69d.pth', 
                                                map_location=f'cpu'), strict=strict)
        
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
            model = mobilenetv3_large(prob_dropout=config['dropout']['prob_dropout'])
            if config['model']['pretrained']:
                model.load_state_dict(torch.load('pretrained/mobilenetv3-large-1cd25616.pth', 
                                                map_location=f'cpu'), strict=strict)
        else:
            assert config['model']['model_size'] == 'small'
            model = mobilenetv3_small(width_mult=.75, prob_dropout=config['dropout']['prob_dropout'])
            if config['model']['pretrained']:
                model.load_state_dict(torch.load('pretrained/mobilenetv3-small-0.75-86c972c3.pth', 
                                                map_location=f'cpu'), strict=strict)

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
