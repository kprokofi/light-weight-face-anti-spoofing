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
import inspect
import os.path as osp
import sys

import cv2 as cv
import glog as log
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from demo_tools import TorchCNN, VectorCNN


def main():
    """Prepares data for the accuracy convertation checker"""
    parser = argparse.ArgumentParser(description='antispoofing recognition live demo script')
    parser.add_argument('--config', type=str, default=None, required=True,
                        help='Configuration file')
    parser.add_argument('--spf_model_openvino', type=str, default=None,
                        help='path to .xml IR OpenVINO model', required=True)
    parser.add_argument('--spf_model_torch', type=str, default=None,
                        help='path to .pth.tar checkpoint', required=True)
    parser.add_argument('--device', type=str, default='CPU')

    args = parser.parse_args()
    config = utils.read_py_config(args.config)
    assert args.spf_model_openvino.endswith('.xml') and args.spf_model_torch.endswith('.pth.tar')
    spoof_model_torch = utils.build_model(config, args.device.lower(), strict=True, mode='eval')
    spoof_model_torch = TorchCNN(spoof_model_torch, args.spf_model_torch, config, device=args.device.lower())
    spoof_model_openvino = VectorCNN(args.spf_model_openvino)
    # running checker
    avg_diff = run(spoof_model_torch, spoof_model_openvino)
    print((f'mean difference on the first predicted class : {avg_diff[0]}\n'
           + f'mean difference on the second predicted class : {avg_diff[1]}'))

def pred_spoof(batch, spoof_model_torch, spoof_model_openvino):
    """Get prediction for all detected faces on the frame"""
    output1 = spoof_model_torch.forward(batch)
    output1 = list(map(lambda x: x.reshape(-1), output1))
    output2 = spoof_model_openvino.forward(batch)
    output2 = list(map(lambda x: x.reshape(-1), output2))
    return output1, output2

def check_accuracy(torch_pred, openvino_pred):
    diff = np.abs(np.array(openvino_pred) - np.array(torch_pred))
    avg = diff.mean(axis=0)
    return avg

def run(spoof_model_torch, spoof_model_openvino):
    batch = np.float32(np.random.rand(100,128,128,3))
    torch_pred, openvino_pred = pred_spoof(batch, spoof_model_torch, spoof_model_openvino)
    avg_diff = check_accuracy(torch_pred, openvino_pred)
    return avg_diff

if __name__ == '__main__':
    main()
