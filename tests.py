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
