import os
import torch
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np


class LCFAD(Dataset):
    def __init__(self, root_dir, protocol='train', transform=None):
        assert protocol in ['train', 'val', 'test', 'combine_partly', 'combine_all']
        self.root_dir = root_dir
        self.transform = transform
        if protocol == 'train':
            spoof_img, real_img = get_train_img(self.root_dir)
            self.list_img = spoof_img + real_img
            self.labels = torch.cat((torch.ones(len(spoof_img), dtype=torch.long),
                                    (torch.zeros(len(real_img), dtype=torch.long))))
        elif protocol == 'val':
            spoof_img, real_img = get_val_img(self.root_dir)
            self.list_img = spoof_img + real_img
            self.labels = torch.cat((torch.ones(len(spoof_img), dtype=torch.long),
                                    (torch.zeros(len(real_img), dtype=torch.long))))
        elif protocol == 'test':
            spoof_img, real_img = get_test_img(self.root_dir)
            self.list_img = spoof_img + real_img
            self.labels = torch.cat((torch.ones(len(spoof_img), dtype=torch.long),
                                        (torch.zeros(len(real_img), dtype=torch.long))))
        elif protocol == 'combine_partly':
            spoof_img_train, real_img_train = get_train_img(self.root_dir)
            spoof_img_val, real_img_val = get_val_img(self.root_dir)
            self.list_img = spoof_img_train + spoof_img_val + real_img_train + real_img_val
            self.labels = torch.cat((torch.ones(len(spoof_img_train) + len(spoof_img_val), dtype=torch.long),
                                        (torch.zeros(len(real_img_train) + len(real_img_val), dtype=torch.long))))
        else:
            spoof_img_train, real_img_train = get_train_img(self.root_dir)
            spoof_img_val, real_img_val = get_val_img(self.root_dir)
            spoof_img_test, real_img_test = get_test_img(self.root_dir)
            self.list_img = spoof_img_train + spoof_img_val + spoof_img_test + real_img_train + real_img_val + real_img_test
            self.labels = torch.cat((torch.ones(len(spoof_img_train) + len(spoof_img_val) + len(spoof_img_test), dtype=torch.long),
                                        (torch.zeros(len(real_img_train) + len(real_img_val) + len(real_img_test), dtype=torch.long))))
    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.list_img[index])
        image = cv.imread(img_path, flags=1)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        y_label = self.labels[index]
        if self.transform:
            image = self.transform(label=y_label, img=image)['image']
        # [batch, channels, height, width]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return (torch.tensor(image), y_label)

def get_val_img(root_dir):
    name_of_real_img = filter(lambda x: x.endswith('.png') or x.endswith('.jpg'),
                    os.listdir(os.path.join(root_dir, 'LCC_FASD_development/real')))
    real_img = list(map(lambda x: os.path.join('LCC_FASD_development/real', x), name_of_real_img))
    name_of_spoof_img = filter(lambda x: x.endswith('.png') or x.endswith('.jpg'),
                    os.listdir(os.path.join(root_dir, 'LCC_FASD_development/spoof')))
    spoof_img = list(map(lambda x: os.path.join('LCC_FASD_development/spoof', x), name_of_spoof_img))
    return spoof_img, real_img

def get_train_img(root_dir):
    name_of_real_img = filter(lambda x: x.endswith('.png') or x.endswith('.jpg'), 
                    os.listdir(os.path.join(root_dir, 'LCC_FASD_training/real')))
    real_img = list(map(lambda x: os.path.join('LCC_FASD_training/real', x), name_of_real_img))

    name_of_spoof_img = filter(lambda x: x.endswith('.png') or x.endswith('.jpg'),
                    os.listdir(os.path.join(root_dir, 'LCC_FASD_training/spoof')))
    spoof_img = list(map(lambda x: os.path.join('LCC_FASD_training/spoof', x), name_of_spoof_img))
    return spoof_img, real_img

def get_test_img(root_dir):
    name_of_real_img = filter(lambda x: x.startswith('real'), 
                    os.listdir(os.path.join(root_dir, 'LCC_FASD_evaluation')))
    real_img = list(map(lambda x: os.path.join('LCC_FASD_evaluation', x), name_of_real_img))

    name_of_spoof_img = filter(lambda x: x.startswith('spoof'), 
                        os.listdir(os.path.join(root_dir, 'LCC_FASD_evaluation')))
    spoof_img = list(map(lambda x: os.path.join('LCC_FASD_evaluation', x), name_of_spoof_img))
    return spoof_img, real_img
