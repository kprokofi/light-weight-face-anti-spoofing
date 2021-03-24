'''MIT License
Copyright (C) 2020 Prokofiev Kirill, Intel Corporation
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.'''

import argparse
import os.path as osp
import shutil

import cv2 as cv
from tqdm import tqdm

from datasets import LccFasdDataset
from demo.demo import FaceDetector


def main():
    """Prepares data for the antispoofing recognition demo"""
    # arguments parcing
    parser = argparse.ArgumentParser(description='prepare LCC FASD')
    parser.add_argument('--fd_model', type=str, required=True, help='path to fd model')
    parser.add_argument('--fd_thresh', type=float, default=0.6, help='Threshold for FD')
    parser.add_argument('--device', type=str, default='CPU')
    parser.add_argument('--root_dir', type=str, required=True, help='LCC FASD root dir')
    parser = argparse.ArgumentParser(description='LCC_FASD')
    args = parser.parse_args()
    face_detector = FaceDetector(args.fd_model, args.fd_thresh, args.device)
    protocols = ['train', 'val', 'test']
    print('===> processing the data...')
    save_dir = osp.abspath(shutil.copytree(args.root_dir, './LCC_FASDcropped',
                           ignore=shutil.ignore_patterns('*.png', '.*')))
    dir_path = osp.abspath(args.root_dir)
    for protocol in protocols:
        data =  LccFasdDataset(root_dir=args.root_dir, protocol=protocol,
                      transform=None, get_img_path=True)
        for image, path in tqdm(data, desc=protocol, total=len(data), leave=False):
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
