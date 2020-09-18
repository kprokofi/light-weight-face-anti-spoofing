'''MIT License

Copyright (C) 2020 Prokofiev Kirill
 
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
import os
import os.path as osp
import glog as log
import cv2 as cv
import numpy as np
from scipy.spatial.distance import cosine
from ie_tools import load_ie_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import utils

class FaceDetector:
    """Wrapper class for face detector"""
    def __init__(self, model_path, conf=.6, device='CPU', ext_path=''):
        self.net = load_ie_model(model_path, device, None, ext_path)
        self.confidence = conf
        self.expand_ratio = (1.1, 1.05)

    def get_detections(self, frame):
        """Returns all detections on frame"""
        _, _, h, w = self.net.get_input_shape().shape
        out = self.net.forward(cv.resize(frame, (w, h)))
        detections = self.__decode_detections(out, frame.shape)
        return detections

    def __decode_detections(self, out, frame_shape):
        """Decodes raw SSD output"""
        detections = []

        for detection in out[0, 0]:
            confidence = detection[2]
            if confidence > self.confidence:
                left = int(max(detection[3], 0) * frame_shape[1])
                top = int(max(detection[4], 0) * frame_shape[0])
                right = int(max(detection[5], 0) * frame_shape[1])
                bottom = int(max(detection[6], 0) * frame_shape[0])
                if self.expand_ratio != (1., 1.):
                    w = (right - left)
                    h = (bottom - top)
                    dw = w * (self.expand_ratio[0] - 1.) / 2
                    dh = h * (self.expand_ratio[1] - 1.) / 2
                    left = max(int(left - dw), 0)
                    right = int(right + dw)
                    top = max(int(top - dh), 0)
                    bottom = int(bottom + dh)

                detections.append(((left, top, right, bottom), confidence))

        if len(detections) > 1:
            detections.sort(key=lambda x: x[1], reverse=True)
        return detections

class VectorCNN:
    """Wrapper class for a nework returning a vector"""
    def __init__(self, model_path, device='CPU'):
        self.net = load_ie_model(model_path, device, None)

    def forward(self, batch):
        """Performs forward of the underlying network on a given batch"""
        _, _, h, w = self.net.get_input_shape().shape
        outputs = [self.net.forward(cv.resize(frame, (w, h))) for frame in batch]
        return outputs

class TorchCNN:
    '''Wrapper for torch model'''
    def __init__(self, model, checkpoint_path, config, device='cpu'):
        self.model = model
        if config.data_parallel.use_parallel.:
            self.model = nn.DataParallel(self.model, **config.data_parallel.parallel_params.)
        utils.load_checkpoint(checkpoint_path, self.model, map_location=device)
        self.config = config

    def preprocessing(self, images):
        ''' making image preprocessing for pytorch pipeline '''
        mean = np.array(self.config.img_norm_cfg.mean.).reshape(3,1,1)
        std = np.array(self.config.img_norm_cfg.std.).reshape(3,1,1)
        height, width = list(self.config.resize..values())
        for i in range(len(images)):
            images[i] = cv.resize(images[i], (height, width) , interpolation=cv.INTER_CUBIC)
            images[i] = cv.cvtColor(images[i], cv.COLOR_BGR2RGB)
            images[i] = np.transpose(images[i], (2, 0, 1)).astype(np.float32)
            images[i] = images[i]/255
            images[i] = (images[i] - mean)/std
        return torch.tensor(images, dtype=torch.float32)

    def forward(self, batch):
        batch = self.preprocessing(batch)
        self.model.eval()
        if self.config.data_parallel.use_parallel.:
            model1 = self.model.module
        else:
            model1 = self.model
        with torch.no_grad():
            features = model1.forward(batch)
            output = model1.spoof_task(features)
            if type(output) == tuple:
                output = output[0]
            confidence = F.softmax(output, dim=-1).detach().numpy()
            return confidence

def pred_spoof(frame, detections, spoof_model, config):
    """Get prediction for all detected faces on the frame"""
    faces = []
    for rect, _ in detections:
        left, top, right, bottom = rect
        # cut face according coordinates of detections
        faces.append(frame[top:bottom, left:right])
    if faces:
        output = spoof_model.forward(faces)
        output = list(map(lambda x: x.reshape(-1), output))
        return output
    return None, None

def draw_detections(frame, detections, confidence):
    """Draws detections and labels"""
    for i, rect in enumerate(detections):
        left, top, right, bottom = rect[0]
        if confidence[i][1] > 0.4:
            label = f'spoof: {round(confidence[i][1]*100, 3)}%'
            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
        else:
            label = f'real: {round(confidence[i][0]*100, 3)}%'
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness=2)
        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 1)
        top = max(top, label_size[1])
        cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                     (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    return frame

def run(params, capture, face_det, spoof_model, config):
    """Starts the anti spoofing demo"""
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    writer_video = cv.VideoWriter('output_video.mp4', fourcc, 24, (720,540))
    win_name = 'Antispoofing Recognition'
    while cv.waitKey(1) != 27:
        has_frame, frame = capture.read()
        if not has_frame:
            return
        detections = face_det.get_detections(frame)
        confidence = pred_spoof(frame, detections, spoof_model, config)
        frame = draw_detections(frame, detections, confidence)
        cv.imshow(win_name, frame)
        writer_video.write(cv.resize(frame, (720,540)))

def load_checkpoint(checkpoint, model):
    print("==> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])

def main():
    """Prepares data for the antispoofing recognition demo"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='antispoofing recognition live demo script')
    parser.add_argument('--video', type=str, default=None, help='Input video')
    parser.add_argument('--cam_id', type=int, default=-1, help='Input cam')
    parser.add_argument('--config', type=str, default=None, required=True,
                        help='Configuration file')
    parser.add_argument('--fd_model', type=str, required=True)
    parser.add_argument('--fd_thresh', type=float, default=0.6, help='Threshold for FD')

    parser.add_argument('--spf_model', type=str, default=None, help='path to .pth checkpoint of model or .xml IR OpenVINO model', required=True)

    parser.add_argument('--device', type=str, default='CPU')
    parser.add_argument('-l', '--cpu_extension',
                        help='MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels '
                             'impl.', type=str, default=None)

    args = parser.parse_args()
    config = utils.read_py_config(args.config)

    if args.cam_id >= 0:
        log.info('Reading from cam {}'.format(args.cam_id))
        cap = cv.VideoCapture(args.cam_id)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    else:
        assert args.video
        log.info('Reading from {}'.format(args.video))
        cap = cv.VideoCapture(args.video)
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        
    assert cap.isOpened()

    face_detector = FaceDetector(args.fd_model, args.fd_thresh, args.device, args.cpu_extension)
    if args.spf_model.endswith('pth.tar'):
        config.model.pretrained. = False
        spoof_model = utils.build_model(config, args, strict=True)
        spoof_model = TorchCNN(spoof_model, args.spf_model, config)
    else:
        spoof_model = VectorCNN(args.spf_model)
    
    run(args, cap, face_detector, spoof_model, config)

if __name__ == '__main__':
    main()