'''MIT License

Copyright (C) 2019-2020 Intel Corporation
 
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

from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import albumentations as A
import torch
import numpy as np
import argparse
import os
from utils import make_loader, make_dataset, load_checkpoint, build_model, read_py_config, Transform, make_output
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import torch.nn as nn
from tqdm import tqdm
import cv2 as cv

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='antispoofing training')
    parser.add_argument('--draw_graph', default=False, type=bool, required=False, help='whether or not to draw graphics')
    parser.add_argument('--GPU', default=0, type=int, required=False, help='specify which GPU to use')
    parser.add_argument('--config', type=str, default=None, required=True,
                        help='path to configuration file')
    args = parser.parse_args()

    path_to_config = args.config
    config = read_py_config(path_to_config)
    config['model']['pretrained'] = False
    model = build_model(config, args, strict=True)
    model.cuda(device=args.GPU)
    if config['data_parallel']['use_parallel']:
        model = torch.nn.DataParallel(model, **config['data_parallel']['parallel_params'])
    # load snapshot
    path_to_experiment = os.path.join(config['checkpoint']['experiment_path'], config['checkpoint']['snapshot_name'])
    load_checkpoint(path_to_experiment, model, map_location=torch.device(f'cuda:{args.GPU}'), optimizer=None)
    epoch_of_checkpoint = checkpoint['epoch']
    # preprocessing
    normalize = A.Normalize(**config['img_norm_cfg'])
    test_transform = A.Compose([
                A.Resize(**config['resize'], interpolation=cv.INTER_CUBIC),
                normalize,
                ])  
    # making dataset and loader
    test_transform = Transform(val=test_transform)
    test_dataset = make_dataset(config, val_transform=test_transform, mode='eval')
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True, num_workers=2)
    # computing metrics
    AUC, EER, accur, apcer, bpcer, acer, fpr, tpr  = evaulate(model, test_loader, config, args, compute_accuracy=True)

    print(f'EER = {round(EER*100,2)}\n\
    accuracy on test data = {round(np.mean(accur),3)}\n\
    AUC = {round(AUC,3)}\n\
    apcer = {round(apcer*100,2)}\n\
    bpcer = {round(bpcer*100,2)}\n\
    acer = {round(acer*100,2)}\n\
    checkpoint made on {epoch_of_checkpoint} epoch')
 
    if args.draw_graph:
        fnr = 1 - tpr
        plot_ROC_curve(fpr, tpr, config)
        DETCurve(fpr, fnr, EER, config)

def evaulate(model, loader, config, args, compute_accuracy=True):
    ''' evaulating AUC, EER, ACER, BPCER, APCER on given data loader and model '''
    model.eval()
    proba_accum = np.array([])
    target_accum = np.array([])
    accur=[]
    tp, tn, fp, fn = 0, 0, 0, 0
    for input, target in tqdm(loader):
        input = input.cuda(device=args.GPU)
        if len(target.shape) > 1:
            target = target[:, 0].reshape(-1).cuda(device=args.GPU)
        with torch.no_grad():
            features = model(input)
            if config['data_parallel']['use_parallel']:
                model1 = model.module
            else:
                model1 = model
            output = model1.spoof_task(features)
            if type(output)==tuple:
                output = output[0]

            y_true = target.detach().cpu().numpy()
            y_pred = output.argmax(dim=1).detach().cpu().numpy()
            tn_batch, fp_batch, fn_batch, tp_batch = metrics.confusion_matrix(y_true=y_true, 
                                                                              y_pred=y_pred, 
                                                                              ).ravel()
            
            tp += tp_batch
            tn += tn_batch
            fp += fp_batch
            fn += fn_batch

            if compute_accuracy:
                accur.append((y_pred == y_true).mean())
            if config['loss']['loss_type'] == 'amsoftmax':
                output *= config['loss']['amsoftmax']['s']
            if config['loss']['loss_type'] == 'soft_triple':
                output *= config['loss']['soft_triple']['s']
            positive_probabilities = F.softmax(output, dim=-1)[:,1].cpu().numpy()
        proba_accum = np.concatenate((proba_accum, positive_probabilities))
        target_accum = np.concatenate((target_accum, y_true))

    apcer = fp / (tn + fp) if fp != 0 else 0
    bpcer = fn / (fn + tp) if fn != 0 else 0
    acer = (apcer + bpcer) / 2

    fpr, tpr, _ = roc_curve(target_accum, proba_accum, pos_label=1)
    fnr = 1 - tpr
    fpr_EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    fnr_EER = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    if fpr_EER < fnr_EER:
        EER = fpr_EER
    else:
        EER = fnr_EER
    AUC = auc(fpr, tpr)
    if compute_accuracy:
        return AUC, EER, accur, apcer, bpcer, acer, fpr, tpr
    return AUC, EER, accur, apcer, bpcer, acer

def plot_ROC_curve(fpr, tpr, config):
    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.00])
    plt.plot(fpr, tpr, lw=3, label="ROC curve (area= {:0.2f})".format(auc(fpr, tpr)))
    plt.xlabel('FPR', fontsize=16)
    plt.ylabel('TPR', fontsize=16)
    plt.title('ROC curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0,1],[0,1], lw=3, linestyle='--', color='navy')
    plt.axes().set_aspect('equal')
    plt.savefig(config['curves']['det_curve'])

def DETCurve(fps,fns, EER, config):
    """
    Given false positive and false negative rates, produce a DET Curve.
    The false positive rate is assumed to be increasing while the false
    negative rate is assumed to be decreasing.
    """
    fig,ax = plt.subplots(figsize=(8,8))
    plt.plot(fps,fns, label=f"DET curve, EER%={round(EER*100, 3)}")
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('FAR', fontsize=16)
    plt.ylabel('FRR', fontsize=16)
    ticks_to_use = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1]
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(ticks_to_use)
    ax.set_yticks(ticks_to_use)
    plt.xticks(rotation=45)
    plt.axis([0.001,1,0.001,1])
    plt.title('DET curve', fontsize=20)
    plt.legend(loc='upper right', fontsize=16)
    fig.savefig(config['curves']['det_curve'])

if __name__ == "__main__":
    main()
    
