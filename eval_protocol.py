from sklearn.metrics import roc_curve, auc
from reader_dataset_tmp import LCFAD_test
from datasets.lcc_fasd import LCFAD
from datasets.casia_surf import CasiaSurfDataset
from datasets import CelebASpoofDataset
import albumentations as A
import torch
import numpy as np
import argparse
import os
from utils import make_loader, make_dataset, load_checkpoint, build_model, read_py_config
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import torch.nn as nn
from sklearn import metrics
from tqdm import tqdm

def evaulate(model, loader, config, args, compute_accuracy=True):
    ''' evaulating AUC, EER, ACER, BPCER, APCER on given data loader and model '''
    model.eval()
    proba_accum = np.array([])
    target_accum = np.array([])
    accur=[]
    tp, tn, fp, fn = 0, 0, 0, 0
    for input, target in tqdm(loader):
        input = input.cuda(device=args.GPU)
        target = target.cuda(device=args.GPU)
        # target = 1 - target
        with torch.no_grad():
            output = model(input)
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
            else:
                assert config['loss']['loss_type'] == 'soft_triple'
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
    fig,ax = plt.subplots()
    plt.plot(fps,fns, label=f"DET curve, EER%={round(EER*100, 3)}")
    plt.yscale('log')
    plt.xscale('log')
    ticks_to_use = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50]
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(ticks_to_use)
    ax.set_yticks(ticks_to_use)
    plt.axis([0.001,50,0.001,50])
    fig.savefig(config['curves']['det_curve'])

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='antispoofing training')
    parser.add_argument('--draw_graph', default=False, type=bool, help='whether or not to draw graphics')
    parser.add_argument('--GPU', default=2, type=int, help='whether or not to draw graphics')
    parser.add_argument('--config', type=str, default='config.py', required=True,
                        help='Configuration file')
    args = parser.parse_args()

    path_to_config = os.path.join(current_dir, args.config)
    config = read_py_config(path_to_config)
    model = build_model(config, args, strict=True)
    model.cuda(device=args.GPU)
    # load snapshot
    path_to_experiment = os.path.join(config['checkpoint']['experiment_path'], config['checkpoint']['snapshot_name'])
    checkpoint = torch.load(path_to_experiment, map_location=torch.device(f'cuda:{args.GPU}')) 
    load_checkpoint(checkpoint, model, optimizer=None)
    epoch_of_checkpoint = checkpoint['epoch']
    # preprocessing
    normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = A.Compose([
                A.Resize(224, 224),
                normalize,
                ])  
    # making dataset and loader
    test_dataset = make_dataset(config, val_transform=test_transform, mode='eval')
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True, num_workers=2)
    # computing metrics
    AUC, EER, accur, apcer, bpcer, acer, fpr, tpr  = evaulate(model, test_loader, config, args, compute_accuracy=True)

    print(f'EER = {round(EER*100,2)}')
    print(f'accuracy on test data = {round(np.mean(accur),3)}')
    print(f'AUC = {round(AUC,3)}')
    print(f'apcer = {round(apcer*100,2)}')
    print(f'bpcer = {round(bpcer*100,2)}')
    print(f'acer = {round(acer*100,2)}')
    print(f'checkpoint made on {epoch_of_checkpoint} epoch')

    if args.draw_graph:
        fnr = 1 - tpr
        plot_ROC_curve(fpr, tpr, config)
        DETCurve(fpr, fnr, EER, config)

if __name__ == "__main__":
    main()
    
