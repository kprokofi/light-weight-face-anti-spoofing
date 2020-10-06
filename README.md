# lightweight face anti spoofing
towards the solving anti spoofing problem on RGB only data.
## Introduction
This repository contains training and evaulation pipeline with different regularization methods for face anti-spoofing network. There are a few models available for training purposes, based on MobileNetv2 (MN2) and MobileNetv3 (MN3). Project supports natively three datasets: [CelebA Spoof](https://github.com/Davidzhangyuanhan/CelebA-Spoof), [LCC FASD](https://csit.am/2019/proceedings/PRIP/PRIP3.pdf), [CASIA-SURF CeFA](https://arxiv.org/pdf/2003.05136.pdf). Also you may want to train or validate on your data. Final model based on MN3 trained on CelebA Spoof dataset. Model has 3.83 time less parametrs and 28 time less GFlops than AENET from original paper, in the same time MN3 better generalise on cross domain. The code contains demo which you can launch in real time with your webcam or on provided video. Also, the code supports conversion to the ONNX format.
| model name | dataset | AUC | EER% | APCER% | BPCER% | ACER% | MParam | GFlops | Link to snapshot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MN3_large |CelebA-Spoof| 0.998 | 2.26 | 0.69 | 6.92 | 3.8 | 0.13 | 2.93 | [snapshot](https://drive.google.com/drive/u/0/folders/1A6wa3AlrdjyNPkXT81knIzXxR7SAYm1q) |
| AENET | CelebA-Spoof | 0.999 | 1.12 | 0.23 | 6.27 | 3.25 | 3.64 | 11.22 | [link to repo](https://github.com/Davidzhangyuanhan/CelebA-Spoof) |
| MN3_large | LCC_FASD | 0.921 | 16.13 | 17.26 | 15.4 | 16.33 | 0.13 | 2.93 | [snapshot](https://drive.google.com/drive/u/0/folders/1A6wa3AlrdjyNPkXT81knIzXxR7SAYm1q) |
| AENET | CLCC_FASD | 0.868 | 20.91 | 12.52 | 32.7 | 22.61 | 3.64 | 11.22 | [link to repo](https://github.com/Davidzhangyuanhan/CelebA-Spoof) |

## Setup
### Prerequisites

* Python 3.6.9
* OpenVINO™ 2020 R3 (or newer) with Python API

### Installation

1. Create virtual environment:
```bash
bash init_venv.sh
```

2. Activate virtual environment:
```bash
. venv/bin/activate
```
### Data Preparation
For training or evaluating on CelebA Spoof dataset you need to download dataset (you can do it from their [official repository](https://github.com/Davidzhangyuanhan/CelebA-Spoof)) and then run following script being located in root folder of the project:
```bash
cd /data_preparation/
python prepare_celeba_json.py
```
To train or evaluate on LCC FASD dataset you need to download it (link is available in their paper on [arxiv](https://csit.am/2019/proceedings/PRIP/PRIP3.pdf)). Then you need to get the OpenVINO™ face detector model (choose the one on the goole drive), activate OpenVINO™ enviroment and run following script:
```bash
python prepare_LCC_FASD.py --fd_model <path to `.xml` face detector model> --root_dir <path to root dir of LCC_FASD>
```
This script will cut faces tighter than it is in original dataset and get rid of some garbage crops. For running this script you need to activate OpenVINO™ environment. Refer to official documentation.

You can use LCC FASD without doing this at all, but it seems to enhance performance, so I recommend doing this.
Note that the new folder will be created and named as `<old name>cropped`. So to train or test model with cropped data, please, set path to that new folder, which will be located in the same directory as launched script.

To train or evaluate on CASIA CEFA you just need to download it. Reader for this dataset supports not only RGB modality, but depth and IR too. Nevertheless, it's not the purpose of this project.

If you want to use your own data, the next steps should be done:
1) prepare reader for your dataset.
2) Import reader object to datasets/database.py file. Substitute `do_nothing` with your object in `external_reader=do_nothing` (35 line).
3) in config, write any kwargs for train, val, test protocol. If you do not have test data, you can just add the same parameters as in val.
Example: `external = dict(train=dict(data_root='...', mode='train', whatever=...), val=dict(data_root='...', mode='val', whatever=...), test=dict(...))`

Now you are ready to launch training process!

### Configuration file
The script for training and inference uses a configuration file. This is default [configuration file](./configs/config.py). You need to specify paths to datasets. Training pipeline supports following methods, which you can switch on and tune hyperparams while training:
* **dataset** - this is indicator which dataset you will be using during training. Available options are 'celeba-spoof', 'LCC_FASD', 'Casia', 'multi_dataset', 'external'
* **multi_task_learning** - specify whether or not to train with multitask loss. **It is avaliable for CelebA-Spoof dataset only!**
* **evaulation** - it is flag to perform the assessment at the end of training and write metrics to a file
* **test_dataset** - this is indicator on which dataset you want to test. Options are the same as for dataset parameter
* **external** - parameters for constructing external dataset reader. See Data Preparation section.
* **img_norm_cfg** - parameters for data normalization
* **scheduler** - scheduler for dropping learning rate
* **data.sampler** - if it is True, then will be generated weights for `WeightedRandomSampler` object to uniform distribution of two classes
* **resize** - resize of the image
* **checkpoint** - name of the checkpoint to save and path to experiment folder where checkpoint, tensorboard logs and eval metrics will be keeped
* **loss** - there are available two possible losses: 'amsoftmax' with 'cos','arcos','cross_enropy' margins and 'soft_triple' with different number of inner classes. For more details about this soft triple loss see in [paper](https://arxiv.org/pdf/1909.05235.pdf)
* **loss.amsoftmax.ratio** - there are availablity to use different m for different classes. Ratio is weights on which provided `m` will be devided for specific class. For example ratio = [1,2] means that m for the first class will equal to m, but for the second will equal to m/2
* **loss.amsoftmax.gamma** - if this constant differs from 0 then focal loss will be switched on with the correspodnig gamma
* **For soft triple loss**: `Cn` - number of classes, `K` - number of proxies for each class, `tau` - parameter for regularisation number of proxies
* **model** - there are parameters concern model. `pretrained` means that you want to train with imagenet weights (you can download weights from google drive(https://drive.google.com/drive/u/0/folders/1A6wa3AlrdjyNPkXT81knIzXxR7SAYm1q) and specify the path to it in `imagenet weights` parameter. **model_type** - type of the model, 'Mobilenet3' and 'Mobilenet2' are available. **size** param means the size of the mobilenetv3, there are 'large' and 'small' options. Note that this will change mobilenev3 only. **embeding_dim** - the size of the embeding (vector of features after average pooling)
* **aug** - advanced augmentation, appropriate value for type is 'cutmix' or 'mixup. lambda = BetaDistribution(alpha, beta), cutmix_prob - probability of applying cutmix on image.
* **curves** - you can specify name of the curves, then set option '--draw_graph' to True when evaulate with eval_protocol.py script
* **dropout** - 'bernoulli' and 'gaussian' dropout available with respective parameters
* **data_parallel** - you can train your network on several GPU
* **RSC** - representation self challenging, applied before global average pooling. p, b - quantile and probability applying it on image in current batch
* **conv_cd** - this is option to switch on central difference convolutions instead of vanilla one changing value of theta from 0
* **test_steps** - if you set this parameter for some int number, the algorithm will execute that many iterations for one epoch and stop. This will help you to test all processes (train,val,test)

## Training
To start training create config file based on the default one and run 'train.py':
```bash
python train.py --config <path to config>;
```
For additional parameters you can refer to help adding '--help'. For example, you can specify on which GPU you want to train your model. If for some reasons you want to train on CPU, specify `--device` to 'cpu'. default device is cuda 0.

## Testing
To test your model set 'test_dataset' in config file to one of preferable dataset (available params: 'celeba-spoof', 'LCC_FASD', 'Casia'). Then run script:
```bash
python eval_protocol.py --config <path to config>;
```
default device to do it is cuda 0.

## Convert a PyTorch Model to the OpenVINO™ Format
To convert the obtained model, run the following command:
```bash
python convert_model.py --config <path to config>; --model_path <path to where save the model>;
```
By default, the output model path is 'MobileNetv3.onnx'
Now you obtain '.onnx' format. To obtain OpenVINO™ IR model you should refer to [official documentation](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) for next steps. You do not need to add --mean, --scale, --inverse_input channels options.
To check that there are no mistakes with the conversion you can launch `conversion_checker.py` by writing following command:
```bash
python conversion_checker.py --config <path to config>; --spf_model_torch <path to torch model> --spf_model_openvino <path to openvino model>;
```
You will see mean difference (L1 metric distance) on the first and second predicted class. If it's 10e-6 or less than it's all good.

## Demo
To start demo you need to download one of available OpenVINO™ face detector model. On [google drive](https://drive.google.com/drive/u/0/folders/1A6wa3AlrdjyNPkXT81knIzXxR7SAYm1q) you will see a trained antispoofing model that you can download and run, or choose your own trained model. Use OpenVINO™ format to obtain best perfomance speed, but pytorch format will work as well.

After preparation start demo by running:
```bash
python demo/demo.py --fd_model /path_to_face_detecor.xml --spf_model /path_to_antispoofing_model.xml(.pth.tar) --cam_id 0 --config config.py;
```
Refer to `--help` for additional parameters. If you are using pytorch model then you need to specify training config with `--config` option. To run demo on the video, you should change `--cam_id` on `--video` option and specify your_video.mp4
