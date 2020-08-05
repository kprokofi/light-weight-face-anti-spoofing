import random
import os
import re
import torch
import numpy as np

from tqdm import tqdm
from torch.utils import data
from torch import nn
from sklearn import metrics
from argparse import ArgumentParser
import utils
import numpy as np
import torch
import cv2 as cv

from torch.utils.data import Dataset
import albumentations as A


from utils import read_py_config
import MobilNet2


class CasiaSurfDataset(Dataset):
    def __init__(self, protocol: int, dir: str = 'data/CASIA_SURF', mode: str = 'train', depth=False, ir=False,
                 transform=None):
        self.dir = dir
        self.mode = mode
        submode = {'train': 'train', 'dev': 'dev_ref',
                   'test': 'test_res'}[mode]
        file_name = '4@{}_{}.txt'.format(protocol, submode)
        with open(os.path.join(dir, file_name), 'r') as file:
            lines = file.readlines()
            self.items = []
            for line in lines:
                if self.mode == 'train':
                    img_name, label = tuple(line[:-1].split(' '))
                    self.items.append(
                        (self.get_all_modalities(img_name, depth, ir), label))

                elif self.mode == 'dev':
                    folder_name, label = tuple(line[:-1].split(' '))
                    profile_dir = os.path.join(
                        self.dir, folder_name, 'profile')
                    for file in os.listdir(profile_dir):
                        img_name = os.path.join(folder_name, 'profile', file)
                        self.items.append(
                            (self.get_all_modalities(img_name, depth, ir), label))

                elif self.mode == 'test':
                    folder_name = line[:-1].split(' ')[0]
                    profile_dir = os.path.join(
                        self.dir, folder_name, 'profile')
                    for file in os.listdir(profile_dir):
                        img_name = os.path.join(folder_name, 'profile', file)
                        self.items.append(
                            (self.get_all_modalities(img_name, depth, ir), -1))

        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_names, label = self.items[idx]
        images = []
        for img_name in img_names:
            img_path = os.path.join(self.dir, img_name)
            img = cv.imread(img_path, flags=1)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            if self.transform is not None:
                img = self.transform(image=img)['image']
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)
            images += [torch.tensor(img)]

        return torch.cat(images, dim=0), int(label)

    def get_all_modalities(self, img_path: str, depth: bool = True, ir: bool = True) -> list:
        result = [img_path]
        if depth:
            result += [re.sub('profile', 'depth', img_path)]
        if ir:
            result += [re.sub('profile', 'ir', img_path)]

        return result


def evaluate(dataloader: data.DataLoader, model: nn.Module, visualize: bool = False):
    device = next(model.parameters()).device
    model.eval()
    print("Evaluating...")
    tp, tn, fp, fn = 0, 0, 0, 0
    errors = np.array([], dtype=[('img', torch.Tensor),
                                 ('label', torch.Tensor), ('prob', float)])
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            images, labels = batch
            outputs = model(images.to(device))
            outputs = outputs.cpu()
            tn_batch, fp_batch, fn_batch, tp_batch = metrics.confusion_matrix(y_true=labels,
                                                                              y_pred=torch.max(
                                                                                  outputs.data, 1)[1],
                                                                              labels=[0, 1]).ravel()
            if visualize:
                errors_idx = np.where(torch.max(outputs.data, 1)[1] != labels)
                print(errors_idx)
                errors_imgs = list(
                    zip(images[errors_idx], labels[errors_idx], ))
                print(errors_imgs)
                errors = np.append(errors, errors_imgs)

            tp += tp_batch
            tn += tn_batch
            fp += fp_batch
            fn += fn_batch
    apcer = fp / (tn + fp) if fp != 0 else 0
    bpcer = fn / (fn + tp) if fn != 0 else 0
    acer = (apcer + bpcer) / 2
    if visualize:
        print(errors)
        errors.sort(order='prob')
        errors = np.flip(errors)
        print(errors)
        utils.plot_classes_preds(model, zip(*errors))

    return apcer, bpcer, acer


def main(args, config):
    model = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobilNet2.MobileNetV2(use_amsoftmax=True)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['state_dict'])
    val_transform = A.Compose([
                A.Resize(**config['resize']),
                A.Normalize(**config['img_norm_cfg']),
                ])    
    model.eval()
    with torch.no_grad():
        dataset = CasiaSurfDataset(
            args.protocol, mode='dev', dir=args.data_dir, transform=val_transform)
        dataloader = data.DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        apcer, bpcer, acer = evaluate(dataloader, model, args.visualize)
        print('APCER: {}, BPCER: {}, ACER: {}'.format(apcer, bpcer, acer))


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--protocol', type=int, required=True)
    argparser.add_argument('--data-dir', type=str,
                           default=os.path.join('data', 'CASIA_SURF'))
    argparser.add_argument('--checkpoint', type=str, required=True)
    argparser.add_argument('--config', type=str, required=True)
    argparser.add_argument('--batch_size', type=int, default=1)
    argparser.add_argument('--visualize', type=bool, default=False)
    argparser.add_argument('--num_workers', type=int, default=0)
    args = argparser.parse_args()
    config = read_py_config(args.config)
    main(args, config)