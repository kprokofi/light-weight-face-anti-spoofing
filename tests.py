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

import unittest
import torch

from utils import read_py_config, build_model, check_file_exist


class ExportError(Exception):
    pass


class TestONNXExport(unittest.TestCase):
    def setUp(self):
        config = read_py_config('./configs/config.py')
        self.config = config
        self.model = build_model(config, device='cpu', strict=True, mode='convert')
        self.img_size = tuple(map(int, config.resize.values()))
    def test_export(self):
        try:
            # input to inference model
            dummy_input = torch.rand(size=(1, 3, *(self.img_size)), device='cpu')
            self.model.eval()
            torch.onnx.export(self.model, dummy_input, './mobilenetv3.onnx', verbose=False)
            check_file_exist('./mobilenetv3.onnx')
        except ExportError:
            self.fail("Exception raised while exporting to ONNX")

if __name__ == '__main__':
    unittest.main()
