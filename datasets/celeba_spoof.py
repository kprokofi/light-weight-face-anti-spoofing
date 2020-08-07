import os
import json

import cv2 as cv
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


class CelebASpoofDataset(Dataset):
    def __init__(self, root_folder, test_mode=False, transform=None):
        self.root_folder = root_folder
        if test_mode:
            list_path = os.path.join(root_folder, 'metas/intra_test/items_test.json')
        else:
            list_path = os.path.join(root_folder, 'metas/intra_test/items_train.json')

        with open(list_path, 'r') as f:
            self.data = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        img = cv.imread(data_item['path'])
        bbox = data_item['bbox']

        real_h, real_w, _ = img.shape
        x1 = clamp(int(bbox[0]*(real_w / 224)), 0, real_w)
        y1 = clamp(int(bbox[1]*(real_h / 224)), 0, real_h)
        w1 = int(bbox[2]*(real_w / 224))
        h1 = int(bbox[3]*(real_h / 224))

        cropped_face = img[y1 : clamp(y1 + h1, 0, real_h), x1 : clamp(x1 + w1, 0, real_w), :]
        cropped_face = cv.cvtColor(cropped_face, cv.COLOR_BGR2RGB)
        if self.transform:
            cropped_face = self.transform(image=cropped_face)['image']

        return (torch.tensor(cropped_face), data_item['labels'][42]) #see readme of the CelebA-Spoof to get layout of labels


def clamp(x, min_x, max_x):
    return min(max(x, min_x), max_x)
