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

import torch
import torchvision
from utils import load_checkpoint, build_model, read_py_config
import argparse
import onnx

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='converting model to onnx')
    parser.add_argument('--GPU', type=int, default=0, required=False, help='specify which gpu to use')
    parser.add_argument('--config', type=str, default=None, required=True,
                            help='path to configuration file')
    parser.add_argument('--model_path', type=str, default='MobileNetv3.onnx', required=False,
                            help='path where to save the model in onnx format')
    parser.add_argument('--num_layers', type=int, default=16, required=False,
                            help='number of the layers of your model to create required number of the input names')
    parser.add_argument('--img_size', type=tuple, default=(128,128), required=False,
                        help='height and width of the image to resize')                        
    args = parser.parse_args()
    # read config
    path_to_config = args.config
    config = read_py_config(path_to_config)
    # get snapshot
    experiment_snapshot = config.checkpoint.snapshot_name
    experiment_path = config.checkpoint.experiment_path
    # input to inference model
    dummy_input = torch.randn(1, 3, *args.img_size, device=f'cuda:{args.GPU)}')
    # build model
    model = build_model(config, args, strict=True)
    model.cuda(device=args.GPU)
    # if model trained as data parallel object
    if config.data_parallel.use_parallel:
        model = torch.nn.DataParallel(model, **config.data_parallel.parallel_params)
    # load checkpoint from config
    load_checkpoint(path_to_experiment, model, map_location=torch.device(f'cuda:{args.GPU}'), optimizer=None, strict=True)
    # convert model to onnx
    model.eval()
    input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(args.num_layers) ]
    output_names = [ "output1" ]
    torch.onnx.export(model, dummy_input, args.model_path, verbose=True, input_names=input_names, output_names=output_names)

if __name__=='__main__':
    main()
