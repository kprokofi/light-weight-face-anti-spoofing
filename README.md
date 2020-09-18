# lightweight face anti spoofing
towards the solving anti spoofing problem on RGB only data.
## Introduction
This repository contains training and evaulation pipeline with different regularization methods for face anti-spoofing network. There are two models avaible for training purposes, based on MobileNetv2 (MN2) and MobileNetv3 (MN3). Project contains support for three datasets: [CelebA Spoof](), [LCC FASD](), [Casia CEFA](). Feel free to train and evaulate your model on any dataset. Final model based on MN3 trained on CelebA Spoof dataset. Model has 5 time less parametrs and 24.6 time less GFlops than AENET from original paper, in the same time MN3 better generalise on cross domain. The code contains demo which you can launch in real time with your webcam or on provided video. Also, the code supports conversion to the ONNX format.
| model name | AUC | EER | APCER | BPCER | ACER | MParam | GFlops | Link to snapshot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MN3 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | [snapshot]() |
* all metrics got on CelebA Spoof
## Setup
### Prerequisites

* Python 3.6.9
* torch==1.5.1
* torchvision==0.6.1
* OpenVINO™ 2020 R3 (or newer) with Python API
### Installation

1. Create virtual environment:
```bash
cd /venv_install/
bash init_venv.sh
```

2. Activate virtual environment:
```bash
. venv/bin/activate
```
### Data Preparation
For training or evaluating on CelebA Spoof dataset you need to download dataset (you can do it from their official repository) and then run following script being located in root folder of the project:
```bash
cd /data_preparation/
python3 prepare_celeba_json.py
```
To train or evaluate on LCC FASD dataset you need to download it (link is available in their paper on [arxive]()) and run following script:
```bash
python3 prepare_LCC_FASD.py
```
This script will cut faces tighter than it is in original dataset and get rid of some garbage crops. For running this script you need to activate OpenVINO™ environment. Refer to official documentation.
You can use LCC FASD without doing this at all, but it seems to enhance performance, so I recommend doing this.
To train or evaluate on CASIA CEFA you just need to download it. Reader for this dataset supports not only RGB modality, but depth and IR too. Nevertheless, it's not the purpose of this project.
### Configuration file
The script for training and inference uses a configuration file. This is default [configuration file](). You need to specify paths to datasets. Training pipeline supports following methods, which you can switch on and tune hyperparams while training:
* RSC - representation self challenging, applied before global average pooling. p, b - quantile and probability applying it on image in current batch
* aug - advanced augmentation, appropriate value for type is 'cutmix' or 'mixup. lambda = BetaDistribution(alpha, beta), cutmix_prob - probability of applying cutmix on image.
...
* ...
## Training
To start training create config file based on the default one and run 'train.py':
```bash
sudo python3 train.py --config <path to config>;
```
For additional parameters you can refer to help adding '--help'. For example, you can specify on which GPU you want to train your model. As you may notice, training pipeline supports parallel training and specified GPU in arguments must be the same as in configuration file for output GPU.
## Testing
To test your model set 'test_dataset' in config file to one of preferable dataset (available params: 'celeba-spoof', 'LCC_FASD', 'Casia'). Then run script:
```bash
sudo python3 eval_protocol.py --config <path to config>;
```
## Convert a PyTorch Model to the OpenVINO™ Format
To convert the obtained model, run the following command:
```bash
sudo python3 convert_model.py --config <path to config>; --model_path <path to where save the model>;
```
By default, the output model path is 'model.onnx'
Now you obtain '.onnx' format. To obtain OpenVINO™ IR model you should refer to [official documentation](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) for next steps.
## Demo
Coming soon..
