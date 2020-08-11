import albumentations as A
from tqdm import tqdm
from utils import AverageMeter, read_py_config, save_checkpoint, precision, mixup_target, freeze_layers, make_dataset, make_loader, change_model
import os
from utils import precision, make_dataset, make_loader, change_model
import cv2
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='antispoofing training')
current_dir = os.path.dirname(os.path.abspath(__file__))
parser.add_argument('--GPU', type=int, default=1, help='specify which gpu to use')
parser.add_argument('--print-freq', '-p', default=5, type=int, help='print frequency (default: 20)')
parser.add_argument('--save_checkpoint', type=bool, default=True, help='whether or not to save your model')
parser.add_argument('--config', type=str, default='config.py', required=False,
                        help='Configuration file')

# global variables and argument parsing
args = parser.parse_args()
path_to_config = os.path.join(current_dir, args.config)
config = read_py_config(path_to_config)
i = 0

# load data
train_dataset, val_dataset = make_dataset(config, None, None)
train_loader, val_loader = make_loader(train_dataset, val_dataset, config)

print('\n_________TRAIN____________\n\n\n\n')

train_acc = []
# i = 0
for path, label, shape in train_loader:
    train_acc.append([*path, *label.numpy(), *shape[1].numpy(), *shape[2].numpy()])
    # i+=1
    # if i == 10:
    #     break

train = np.array(train_acc)
train_df = pd.DataFrame(train, columns=['path', 'label', 'height', 'width'])

train_df['spoof'] = train_df['path'].apply(lambda x: 'spoof' in x)
train_df['train'] = train_df['path'].apply(lambda x: 'train' in x)
print('\n\n\n train is train: {}'.format(all(train_df['train'] == True)))
print('\n\n\ncheck contradiction (must be empty): \n{}'.format(train_df[(train_df['spoof'] == True) & (pd.to_numeric(train_df['label']) == 0)]))
print('\n\n\ncheck duplicates (must be true): \n{}'.format(all(train_df['path'].value_counts() == 1))) # check duplicates
print('\n\n\nimbalance: \n{}'.format(train_df['label'].value_counts()))
print('\n\n\nheight of cropped face: \n{}'.format(pd.to_numeric(train_df['height']).describe()))
print('\n\n\nwidth of cropped face: \n{}'.format(pd.to_numeric(train_df['width']).describe()))
print('\n\n\nquantity of train image: \n{}'.format(train_df.shape[0]))

print('\n\n\n_________TEST____________\n\n\n')
# i = 0
test_acc = []
for path, label, shape in val_loader:
    test_acc.append([*path, *label.numpy(), *shape[1].numpy(), *shape[2].numpy()])
    # i+=1
    # if i == 10:
    #     break
test = np.array(test_acc)
test_df = pd.DataFrame(test, columns=['path', 'label', 'height', 'width'])

test_df['spoof'] = test_df['path'].apply(lambda x: 'spoof' in x)
test_df['test'] = test_df['path'].apply(lambda x: 'test' in x)
print('\n\n\n test is test: {}'.format(all(test_df['test'] == True)))
print('\n\n\ncheck contradiction (must be empty): \n{}'.format(test_df[(test_df['spoof'] == True) & (pd.to_numeric(test_df['label']) == 0)]))
print('\n\n\ncheck duplicates (must be true): \n{}'.format(all(test_df['path'].value_counts() == 1))) # check duplicates
print('\n\n\nimbalance: \n{}'.format(test_df['label'].value_counts()))
print('\n\n\nheight of cropped face: \n{}'.format(pd.to_numeric(test_df['height']).describe()))
print('\n\n\nwidth of cropped face: \n{}'.format(pd.to_numeric(test_df['width']).describe()))
print('\n\n\nquantity of test image: \n{}'.format(test_df.shape[0]))