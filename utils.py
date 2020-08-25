import os.path as osp
import os
import sys
import torch
from importlib import import_module
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import LCFAD, CelebASpoofDataset, CasiaSurfDataset, MultiDataset
from torch.utils.data import DataLoader
from losses import AngleSimpleLinear, SoftTripleLinear, AMSoftmaxLoss, SoftTripleLoss
import torch.nn as nn
from models import mobilenetv2, mobilenetv3_large, mobilenetv3_small
import json
from models import Dropout

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

def mixup_target(input, target, config, cuda, num_classes=2):
    # compute mix-up augmentation
    input, target_a, target_b, lam = mixup_data(input, target, config['aug']['alpha'], config['aug']['beta'], cuda)
    input, target_a, target_b = map(Variable, (input, target_a, target_b))
    # compute new target
    target_a_hot = F.one_hot(target_a, num_classes)
    target_b_hot = F.one_hot(target_b, num_classes)
    new_target = lam*target_a_hot + (1-lam)*target_b_hot
    if config['loss']['loss_type'] == 'amsoftmax':
        return input, new_target
    else:
        assert config['loss']['loss_type'] == 'cross_entropy'
        return input, target_a, target_b, lam

def mixup_data(x, y, alpha=1.0, beta=1.0, cuda=0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, beta)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda(device=cuda)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix(input, target, config, args, num_classes=2):
    r = np.random.rand(1)
    if (config['aug']['beta'] > 0) and (config['aug']['alpha'] > 0) and (r < config['aug']['cutmix_prob']):
        # generate mixed sample
        lam = np.random.beta(config['aug']['alpha'] > 0, config['aug']['beta'] > 0)
        rand_index = torch.randperm(input.size()[0]).cuda(device=args.GPU)
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        if config['loss']['loss_type'] in ('cross_enropy', 'soft_triple'):
            target_a = target
            target_b = target[rand_index]
            return input, target_a, target_b, lam
        assert config['loss']['loss_type'] == 'amsoftmax'
        # get one hot target vectors 
        target_a2 = F.one_hot(target, num_classes=num_classes) 
        target_b2 = F.one_hot(target, num_classes=num_classes)[rand_index]
        # do merging classes (cutmix)
        new_target = lam*target_a2 + (1.0 - lam)*target_b2
        return input, new_target
    if config['loss']['loss_type'] == 'amsoftmax':        
        target = F.one_hot(target, num_classes=2)
        return input, target
    assert config['loss']['loss_type'] == 'cross_entropy'
    return input, target, target, 0


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
        test = CelebASpoofDataset(root_folder=config['datasets']['Celeba_root'], test_mode=True, transform=val_transform)
    elif config['dataset'] == 'celeba-spoof':
        train =  CelebASpoofDataset(root_folder=config['data']['data_root'], test_mode=False, transform=train_transform)
        val = CelebASpoofDataset(root_folder=config['data']['data_root'], test_mode=True, transform=val_transform)
        # test = LCFAD(root_dir=config['datasets']['LCCFASD_root'], train=False, transform=val_transform)
        test = val
    elif config['dataset'] == 'Casia':
        train = CasiaSurfDataset(protocol=1, dir=config['data']['data_root'], mode='train', transform=train_transform)
        val = CasiaSurfDataset(protocol=1, dir=config['data']['data_root'], mode='dev', transform=val_transform)
        test = CasiaSurfDataset(protocol=1, dir=config['data']['data_root'], mode='test', transform=val_transform)
    elif config['dataset'] == 'multi_dataset':
        train = MultiDataset(**config['datasets'], train=True, transform=train_transform)
        val = MultiDataset(**config['datasets'], train=False, transform=val_transform)
        # test = LCFAD(root_dir=config['datasets']['LCCFASD_root'], train=False, transform=val_transform)
        test = CelebASpoofDataset(root_folder=config['datasets']['Celeba_root'], test_mode=True, transform=val_transform)
    if mode == 'eval':
        return test
    return train, val

def make_loader(train, val, config, sampler=None):
    ''' make data loader from given train and val dataset
    train, val -> train loader, val loader'''
    if sampler:
        shuffle = False
    else:
        shuffle = True
    train_loader = DataLoader(dataset=train, batch_size=config['data']['batch_size'],
                                                    shuffle=shuffle, pin_memory=config['data']['pin_memory'],
                                                    num_workers=config['data']['data_loader_workers'], sampler=sampler)

    val_loader = DataLoader(dataset=val, batch_size=config['data']['batch_size'],
                                                shuffle=True, pin_memory=config['data']['pin_memory'],
                                                num_workers=config['data']['data_loader_workers'])
    return train_loader, val_loader

def build_model(config, args, strict=True):
    ''' build model and change layers depends on loss type'''

    if config['model']['model_type'] == 'Mobilenet2':
        model = mobilenetv2(prob_dropout=config['dropout']['prob_dropout'])
        if config['model']['pretrained']:
            model.load_state_dict(torch.load('pretrained/mobilenetv2_128x128-fd66a69d.pth', 
                                                map_location=f'cuda:{args.GPU}'), strict=strict)
        
        if config['loss']['loss_type'] == 'amsoftmax':
            model.conv = nn.Sequential(
                        nn.Conv2d(320, config['model']['embeding_dim'], 1, 1, 0, bias=False),
                        nn.Dropout(p=config['dropout']['classifier']),
                        nn.BatchNorm2d(config['model']['embeding_dim']),
                        nn.PReLU()
                    )
            model.classifier = AngleSimpleLinear(config['model']['embeding_dim'], 2)


        elif config['loss']['loss_type'] == 'soft_triple':
            model.conv = nn.Sequential(
                        nn.Conv2d(320, config['model']['embeding_dim'], 1, 1, 0, bias=False),
                        nn.Dropout(p=config['dropout']['classifier']),
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
                                                map_location=f'cuda:{args.GPU}'), strict=strict)
        else:
            assert config['model']['model_size'] == 'small'
            model = mobilenetv3_small(width_mult=.75, prob_dropout=config['dropout']['prob_dropout'])
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
            model.classifier[1] = Dropout(dist=config['dropout']['type'], mu=config['dropout']['mu'], 
                                                        sigma=config['dropout']['sigma'], 
                                                        p=config['dropout']['classifier'])

            model.classifier[2] = nn.BatchNorm1d(config['model']['embeding_dim'])
            model.classifier[4] = AngleSimpleLinear(config['model']['embeding_dim'], 2)
            
        elif config['loss']['loss_type'] == 'cross_entropy':
            model.classifier[1] == nn.Dropout(p=config['dropout']['classifier'])
            model.classifier[4] = nn.Linear(1280, 2)

        else:
            assert config['loss']['loss_type'] == 'soft_triple'
            model.classifier[0] = nn.Linear(exp_size, config['model']['embeding_dim'])
            model.classifier[1] = nn.Dropout(p=config['dropout']['classifier'])
            model.classifier[2] = nn.BatchNorm1d(config['model']['embeding_dim'])
            model.classifier[4] = SoftTripleLinear(config['model']['embeding_dim'], 2, num_proxies=config['loss']['soft_triple']['K'])
    return model

def build_criterion(config, args):
    if config['loss']['loss_type'] == 'amsoftmax':
        criterion = AMSoftmaxLoss(**config['loss']['amsoftmax'], device=args.GPU)
    elif config['loss']['loss_type'] == 'soft_triple':
        criterion = SoftTripleLoss(**config['loss']['soft_triple'])
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion

class Transform():
    """ class to make diferent transform depends on the label """
    def __init__(self, train_spoof=None, train_real=None, val = None):
        self.train_spoof = train_spoof
        self.train_real = train_real
        self.val_transform = val
        if not all((self.train_spoof, self.train_real)):
            self.train = self.train_spoof or self.train_real
            self.transforms_quantity = 1
        else:
            self.transforms_quantity = 2
    def __call__(self, label, img):
        if self.val_transform:
            return self.val_transform(image=img)
        if self.transforms_quantity == 1:
            return self.train(image=img)
        if label:
            return self.train_spoof(image=img)
        else:
            assert label == 0
            return self.train_real(image=img)

def make_weights(config):
    '''load weights for imbalance dataset to list'''
    if config['dataset'] != 'celeba-spoof':
        raise NotImplementedError
    with open(os.path.join(config['data']['data_root'], 'metas/intra_test/items_train.json') , 'r') as f:
        dataset = json.load(f)
    n = 494185
    weights = [0 for i in range(n)]
    keys = list(map(int, list(dataset.keys())))
    keys.sort()
    assert len(keys) == n
    for key in keys:
        label = int(dataset[str(key)]['labels'][43])
        if label:
            weights[int(key)] = 0.1
        else:
            assert label == 0
            weights[int(key)] = 0.204

    assert len(weights) == n
    return weights

def make_output(model, input, target, config):
    ''' target - one hot
    return output 
    If use rsc compute output applying channel-wise rsc method'''
    if config['RSC']['use_rsc']:
        # making features before avg pooling
        features = model.make_features(input)
        # do everything after convolutions layers, strating with avg pooling
        logits = model.make_logits(features)
        if type(logits) == tuple:
            logits = logits[0]
        # take a derivative, make tensor, shape as features, but gradients insted features
        target_logits = torch.sum(logits*target, dim=1)
        gradients = torch.autograd.grad(target_logits, features, grad_outputs=torch.ones_like(target_logits), create_graph=True)
        # gradients = gradients[0] * features # here the same gradients and maybe multiply them with features
        gradients = gradients[0]
        # get value of 1-p quatile
        quantile = torch.tensor(np.quantile(a=gradients.data.cpu().numpy(), q=1-config['RSC']['p'], axis=(1,2,3)), device=input.device)
        quantile = quantile.reshape(input.size(0),1,1,1)
        # create mask
        mask = gradients < quantile
        # element wise product of features and mask, correction for expectition value 
        new_features = (features*mask)/(1-config['RSC']['p'])
        # compute new logits
        new_logits = model.make_logits(new_features)
        if type(new_logits) == tuple:
            new_logits = new_logits[0]
        # compute this operation batch wise
        random_uniform = torch.rand(size=(input.size(0), 1), device=input.device)
        random_mask = random_uniform <= config['RSC']['b']
        output = torch.where(random_mask, new_logits, logits)
        return output
    else:
        assert config['RSC']['use_rsc'] == False
        return model.forward(input)

