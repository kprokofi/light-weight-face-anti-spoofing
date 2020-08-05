import os.path as osp
import sys
import torch
from importlib import import_module
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F
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
