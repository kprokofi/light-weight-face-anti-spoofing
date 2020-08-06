import argparse
import os
import os.path as osp

import glog as log
import cv2 as cv
import numpy as np
from scipy.spatial.distance import cosine
from mobilenet3 import mobilenetv3_large
from ie_tools import load_ie_model
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def pred_spoof(frame, detections, spoof_model):
    """Get prediction for all detected faces on the frame"""
    faces = []

    for rect, _ in detections:
        left, top, right, bottom = rect
        # cut face according coordinates of detections
        faces.append(frame[top:bottom, left:right])

    if faces:
        faces = make_preprocessing(faces)
        spoof_model.eval()
        output = spoof_model(faces)
        confidence = F.softmax(output, dim=-1).detach().numpy()
        predictions = output.argmax(dim=1).detach().numpy()
        assert len(faces) == len(predictions)

    return predictions, confidence

def make_preprocessing(images):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    for i in range(len(images)):
        images[i] = cv.resize(images[i], (224,224))
        images[i] = np.transpose(images[i], (2, 0, 1)).astype(np.float32)
        images[i] = images[i]/255
        images[i] = (images[i] - mean)/std
    return torch.Tensor(images)

def draw_detections(frame, detections, predictions, confidence):
    """Draws detections and labels"""
    for i, rect in enumerate(detections):
        left, top, right, bottom = rect[0]
        if predictions[i] == 1:
            label = f'spoof: {confidence[i][1]*100}%'
            cv.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), thickness=2)
        else:
            assert predictions[i] == 0
            label = f'real: {confidence[i][0]*100}%'
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness=2)
        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 1)
        top = max(top, label_size[1])
        cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                     (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    return frame

def run(params, capture, face_det, spoof_model):
    """Starts the anti spoofing demo"""

    win_name = 'Antispoofing Recognition'
    while cv.waitKey(1) != 27:
        has_frame, frame = capture.read()
        if not has_frame:
            return

        detections = face_det.get_detections(frame)
        spoof_prediction, confidence = pred_spoof(frame, detections, spoof_model)
        frame = draw_detections(frame, detections, spoof_prediction, confidence)
        cv.imshow(win_name, frame)

def load_checkpoint(checkpoint, model):
    print("==> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])

def main():
    """Prepares data for the antispoofing recognition demo"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='antispoofing recognition live demo script')
    parser.add_argument('--video', type=str, default=None, help='Input video')
    parser.add_argument('--cam_id', type=int, default=-1, help='Input cam')

    parser.add_argument('--fd_model', type=str, required=True)
    parser.add_argument('--fd_thresh', type=float, default=0.6, help='Threshold for FD')

    parser.add_argument('--spf_model', type=str, default=os.path.join(current_dir, 'snapshot_MN3.pth.tar'), help='path to checkpoint of model')

    parser.add_argument('--device', type=str, default='CPU')
    parser.add_argument('-l', '--cpu_extension',
                        help='MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels '
                             'impl.', type=str, default=None)

    args = parser.parse_args()

    if args.cam_id >= 0:
        log.info('Reading from cam {}'.format(args.cam_id))
        cap = cv.VideoCapture(args.cam_id)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    else:
        assert args.video
        log.info('Reading from {}'.format(args.video))
        cap = cv.VideoCapture(args.video)
    assert cap.isOpened()

    face_detector = FaceDetector(args.fd_model, args.fd_thresh, args.device, args.cpu_extension)
    spoof_model = mobilenetv3_large()
    spoof_model.classifier[3] = nn.Linear(1280,2)
    load_checkpoint(torch.load(args.spf_model, map_location='cpu'), spoof_model)
    run(args, cap, face_detector, spoof_model)

if __name__ == '__main__':
    main()