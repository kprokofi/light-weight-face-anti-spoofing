import os
import json

import cv2 as cv
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt 

class CelebASpoofDataset(Dataset):
    def __init__(self, root_folder, test_mode=False, transform=None, test_dataset=False):
        self.root_folder = root_folder
        if test_mode:
            list_path = os.path.join(root_folder, 'metas/intra_test/items_test.json')
        else:
            list_path = os.path.join(root_folder, 'metas/intra_test/items_train.json')

        with open(list_path, 'r') as f:
            self.data = json.load(f)

        self.transform = transform
        self.test_dataset = test_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[str(idx)]
        img = cv.imread(os.path.join(self.root_folder, data_item['path']))
        bbox = data_item['bbox']

        real_h, real_w, _ = img.shape
        x1 = clamp(int(bbox[0]*(real_w / 224)), 0, real_w)
        y1 = clamp(int(bbox[1]*(real_h / 224)), 0, real_h)
        w1 = int(bbox[2]*(real_w / 224))
        h1 = int(bbox[3]*(real_h / 224))

        cropped_face = img[y1 : clamp(y1 + h1, 0, real_h), x1 : clamp(x1 + w1, 0, real_w), :]
        cropped_face = cv.cvtColor(cropped_face, cv.COLOR_BGR2RGB)
        # testing dataset and creating json and log file
        if self.test_dataset:
            path = data_item['path']
            label = int(data_item['labels'][43])
            
            if cropped_face.shape[0] < 20 or cropped_face.shape[1] < 20:
                print(path)
            # test_img = cv.resize(cropped_face, (128,128))
            # plt.imsave(f'/home/prokofiev/pytorch/antispoofing/images/{path[-9:]}', arr = test_img, format='png')
            return path, label, cropped_face.shape

        if self.transform:
            cropped_face = self.transform(image=cropped_face)['image']
        cropped_face = np.transpose(cropped_face, (2, 0, 1)).astype(np.float32)
        return (torch.tensor(cropped_face), int(data_item['labels'][43])) #see readme of the CelebA-Spoof to get layout of labels

def clamp(x, min_x, max_x):
    return min(max(x, min_x), max_x)