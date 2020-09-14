import torch
from datasets import CelebASpoofDataset
from torch.utils.data import DataLoader
from utils import Transform
import albumentations as A
import cv2
from tqdm import tqdm

transforms = A.Compose([
                            A.Resize(*(128,128), interpolation=cv2.INTER_CUBIC),
                            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
                            ])

train_dataset = CelebASpoofDataset('/home/prokofiev/pytorch/antispoofing/CelebA_Spoof', test_mode=False, transform=Transform(transforms), multi_learning=False)
dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)

def compute_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5

    return mean, std

mean, std = compute_mean_std(dataloader)
print(mean, std)