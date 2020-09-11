import torch
import torchvision
from utils import load_checkpoint, build_model, read_py_config,
import onnx

# parse arguments
parser.add_argument('--GPU', type=int, default=0, help='specify which gpu to use')
parser.add_argument('--config', type=str, default='config.py', required=True,
                        help='Configuration file')
parser.add_argument('--model_path', type=str, default='MobileNetv3.onnx', required=False,
                        help='path to save the model in onnx format')
# argument parsing and reading config
args = parser.parse_args()
path_to_config = os.path.join(current_dir, args.config)
config = read_py_config(path_to_config)
experiment_snapshot = config['checkpoint']['snapshot_name']
experiment_path = config['checkpoint']['experiment_path']
# input to inference model
dummy_input = torch.randn(1, 3, 128, 128, device='cuda:0')
# build model
model = build_model(config, args, strict=True)
model.cuda(device=args.GPU)
if config['data_parallel']['use_parallel']:
    model = torch.nn.DataParallel(model, **config['data_parallel']['parallel_params'])
# load checkpoint from config
checkpoint = torch.load(path_to_experiment, map_location=torch.device(f'cuda:{args.GPU}')) 
load_checkpoint(checkpoint['state_dict'], model, optimizer=None, strict=True)
# convert model to onnx
model.eval()
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]
torch.onnx.export(model, dummy_input, args.model_path, verbose=True, input_names=input_names, output_names=output_names)

