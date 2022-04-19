import torch

from simple_IO.config_parser import create_config
from simple_IO.tools.init_tool import init_all
from simple_IO.tools.serve_tool import serve, serve_one

def do_serve(message):
    args_config = 'simple_IO/bert.config'
    args_gpu = 'GPU-7444f9b3-c077-071d-8496-422656b95fe8'
    args_checkpoint = 'simple_IO/model/ljp/LJPBertExercise/checkpoint_9.pkl'

    config = create_config(args_config)

    gpu_list = []

    if args_gpu is not None:
        device_list = args_gpu.split(',')

        for device in range(0, len(device_list)):
            gpu_list.append(int(device))

    is_cuda = torch.cuda.is_available()

    if not is_cuda and len(gpu_list) > 0:
        raise NotImplementedError

    parameters = init_all(config, gpu_list, args_checkpoint)

    # serve(parameters, config, gpu_list)

    return serve_one(parameters, config, gpu_list, message)
