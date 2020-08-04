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
from mobilenetv3 import mobilenetv3_large
from amsoftmax import AngleSimpleLinear

def load_checkpoint(checkpoint, model):
    print("==> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])

def evaulate(model, loader, compute_accuracy=True, GPU=2):
    global args

    model.eval()
    proba_accum = np.array([])
    target_accum = np.array([])
    accur=[]
    for input, target in loader:
        input = input.cuda(device=GPU)
        target = target.cuda(device=GPU)
        with torch.no_grad():
            output = model(input)
            if compute_accuracy:
                accur.append((output.argmax(dim=1) == target).float().mean().item())
            positive_probabilities = F.softmax(output, dim=-1)[:,1].cpu().numpy()
        proba_accum = np.concatenate((proba_accum, positive_probabilities))
        target = target.cpu().numpy()
        target_accum = np.concatenate((target_accum, target))

    fpr, tpr, threshold = roc_curve(target_accum, proba_accum, pos_label=1)
    fnr = 1 - tpr
    fpr_EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    fnr_EER = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    if fpr_EER < fnr_EER:
        EER = fpr_EER
    else:
        EER = fnr_EER
    AUC = auc(fpr, tpr)
    return AUC, EER, accur, fpr, tpr

def plot_ROC_curve(fpr, tpr, name_fig='ROC curve 8'):
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

def plot_curve_DET(fpr, fnr, EER, name_fig='DET curve 8'):
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
    fig.savefig('log_DET_8.png')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='antispoofing training')
    parser.add_argument('--model_name', default='/home/prokofiev/pytorch/antispoofing/log_tensorboard/MobileNet_LCFAD_8/my_best_modelMobileNet_8.pth.tar', type=str)
    parser.add_argument('--draw_graph', default=False, type=bool, help='whether or not to draw graphics')
    parser.add_argument('--model', type=str, default='mobilenet2', help='which model to use')
    args = parser.parse_args()
    if args.model == 'mobilenet2':
        model = MobilNet2.MobileNetV2(use_amsoftmax=False) # add variability to the models
    else:
        model = mobilenetv3_large()
        model.classifier[3] = AngleSimpleLinear(1280, 2)
    model.cuda(device=2)
    normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = A.Compose([
                A.Resize(224, 224),
                normalize,
                ])     

    test_dataset = LCFAD_test(root_dir='/home/prokofiev/pytorch/LCC_FASD', transform=test_transform)

    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True, num_workers=2)

    load_checkpoint(torch.load(args.model_name, map_location='cuda:2'), model)
    AUC, EER, accur, fpr, tpr = evaulate(model, test_loader)
    print(f'EER = {EER}')
    print(f'accuracy on test data = {np.mean(accur)}')
    print(f'AUC = {AUC}')
    if args.draw_graph:
        fnr = 1 - tpr
        plot_ROC_curve(fpr, tpr)
        plot_curve_DET(fpr, fnr, EER)
        DETCurve(fpr, fnr, EER)
    