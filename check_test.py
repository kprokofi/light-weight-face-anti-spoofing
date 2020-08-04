from sklearn.metrics import roc_curve, auc
import MobilNet2 
from reader_dataset_tmp import LCFAD_test
from reader_dataset import LCFAD
import albumentations as A
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib

model = MobilNet2.MobileNetV2(use_amsoftmax=False)
parser = argparse.ArgumentParser(description='antispoofing training')
parser.add_argument('--model_name', default='/home/prokofiev/pytorch/antispoofing/log_tensorboard/MobileNet_LCFAD_1.5/my_best_modelMobileNet2_1.5.pth.tar', type=str)

def load_checkpoint(checkpoint, model):
    print("==> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])

def evaulate(model):
    global args
    args = parser.parse_args()
    model.cuda(device=2)
    normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = A.Compose([
                A.Resize(224, 224),
                normalize,
                ])     

    test_dataset = LCFAD(root_dir='/home/prokofiev/pytorch/LCC_FASD', train=False, transform=test_transform)

    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True, num_workers=2)

    load_checkpoint(torch.load(args.model_name, map_location='cuda:2'), model)
    
    model.eval()
    proba_accum = np.array([])
    target_accum = np.array([])
    accur=[]
    for input, target in test_loader:
        input = input.cuda(device=2)
        target = target.cuda(device=2)
        with torch.no_grad():
            output = model(input)
            accur.append((output.argmax(dim=1) == target).float().mean().item())
            positive_probabilities = F.softmax(output, dim=-1)[:,1].cpu().numpy()
        proba_accum = np.concatenate((proba_accum, positive_probabilities))
        target = target.cpu().numpy()
        target_accum = np.concatenate((target_accum, target))
    return target_accum, proba_accum, accur

true_target, output_prediction, accur = evaulate(model)
fpr, tpr, threshold = roc_curve(true_target, output_prediction, pos_label=1)
fnr = 1 - tpr
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print(f'EER = {EER}')
print(f'accuracy on test data = {np.mean(accur)}')
print(f'AUC = {auc(fpr, tpr)}')
def plot_ROC_curve(fpr, tpr, name_fig='ROC curve 5_5'):
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
    plt.savefig(name_fig)

def plot_curve_DET(fpr, fnr, EER, name_fig='DET curve 5_5'):
    plt.figure()
    plt.xlim([0.1, 60])
    plt.ylim([0.1, 60])
    plt.plot(fpr*100, fnr*100, lw=3, label=f"DET curve, EER%={round(EER*100, 3)}")
    plt.xlabel('FRR%', fontsize=16)
    plt.ylabel('FAR%', fontsize=16)
    plt.title('DET curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.axes().set_aspect('equal')
    plt.savefig(name_fig)

def DETCurve(fps,fns, EER):
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
    fig.savefig('log_DET_5_5.png')

# plot_ROC_curve(fpr, tpr)
# plot_curve_DET(fpr, fnr, EER)
# DETCurve(fpr, fnr, EER)

eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, threshold)(eer)
print(eer, thresh)
# true_positives = ((array_saver == 1) == (target_saver == 1)).sum()
# false_positives = ((array_saver == 1) == (target_saver == 0)).sum()
