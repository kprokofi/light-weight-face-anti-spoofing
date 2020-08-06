import os
import torch
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np


class LCFAD(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        if train:
            name_of_real_img = filter(lambda x: x.endswith('.png'),
                    os.listdir(os.path.join(root_dir, 'LCC_FASD_training/real')))
            real_img = list(map(lambda x: os.path.join('LCC_FASD_training/real', x), name_of_real_img))

            name_of_spoof_img = filter(lambda x: x.endswith('.png'),
                    os.listdir(os.path.join(root_dir, 'LCC_FASD_training/spoof')))
            spoof_img = list(map(lambda x: os.path.join('LCC_FASD_training/spoof', x), name_of_spoof_img))

            self.list_img = spoof_img + real_img
            self.labels = torch.cat((torch.ones(len(spoof_img), dtype=torch.long),
                                    (torch.zeros(len(real_img), dtype=torch.long))))
        else:
            assert train == False
            name_of_real_img = filter(lambda x: x.endswith('.png'),
                    os.listdir(os.path.join(root_dir, 'LCC_FASD_development/real')))
            real_img = list(map(lambda x: os.path.join('LCC_FASD_development/real', x), name_of_real_img))
            name_of_spoof_img = filter(lambda x: x.endswith('.png'),
                    os.listdir(os.path.join(root_dir, 'LCC_FASD_development/spoof')))
            spoof_img = list(map(lambda x: os.path.join('LCC_FASD_development/spoof', x), name_of_spoof_img))
            self.list_img = spoof_img + real_img
            self.labels = torch.cat((torch.ones(len(spoof_img), dtype=torch.long),
                                    (torch.zeros(len(real_img), dtype=torch.long))))
    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.list_img[index])
        image = cv.imread(img_path, flags=1)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        y_label = self.labels[index]
        return (torch.tensor(image), y_label)
