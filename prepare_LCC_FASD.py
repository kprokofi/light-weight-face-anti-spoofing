import argparse
import os
import os.path as osp

import glog as log
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from openvino.inference_engine import IENetwork, IEPlugin
from antispoofing.datasets import LCFAD
from tqdm import tqdm
from demo.ie_tools import IEModel, load_ie_model
from demo.demo import FaceDetector
import shutil

def main():
     """Prepares data for the antispoofing recognition demo"""
    parser = argparse.ArgumentParser(description='prepare LCC FASD')
    parser.add_argument('--fd_model', type=str, required=True, help='path to fd model')
    parser.add_argument('--fd_thresh', type=float, default=0.6, help='Threshold for FD')
    parser.add_argument('--device', type=str, default='CPU')
    parser.add_argument('--root_dir', type=str, required=True, help='LCC FASD root dir')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='LCC_FASD')
    args = parser.parse_args()
    
    face_detector = FaceDetector(args.fd_model, args.fd_thresh, args.device)
    
    protocols = ['train', 'val', 'test']
    print('===> processing the data...')
    save_dir = os.path.abspath(shutil.copytree(args.root_dir, './LCC_FASDcropped', ignore=shutil.ignore_patterns('*.png', '.*')))
    dir_path = os.path.abspath(args.root_dir)
    for protocol in protocols:
        data =  LCFAD(root_dir=args.root_dir, protocol=protocol, transform=None, get_img_path=True)
        for i, (image, path) in tqdm(enumerate(data), desc=protocol, total=len(data), leave=False):
            if image.any():
                detection = face_detector.get_detections(image)
                if detection:
                    rect, _ = detection[0]
                    left, top, right, bottom = rect
                    image1=image[top:bottom, left:right] 
                    if not image1.any():
                        print(f'bad crop, {path}')
                    else:
                        new_path = path.replace(dir_path, save_dir) 
                        cv.imwrite(new_path, image1)
                else:
                    print(f'bad crop, {path}')

if __name__ == "__main__":
    main()