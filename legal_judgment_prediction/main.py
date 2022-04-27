import logging
import sys
import argparse
import configparser
import torch

from legal_judgment_prediction.tools.initialize import init_all
from legal_judgment_prediction.tools.train import train
from legal_judgment_prediction.tools.eval import eval
from legal_judgment_prediction.tools.serve import serve_socket, serve_simple_IO


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

fileHandler = logging.FileHandler('log/Bert.log', mode='a',encoding='utf-8')
fileHandler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(ch)
logger.addHandler(fileHandler)

logger.info(' '.join(sys.argv))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', '-c', help='the path of config file', required=True)
    parser.add_argument('--gpu', '-g', help='the list of gpu IDs', required=True)
    parser.add_argument('--mode', '-m', help='train, eval or serve', required=True)
    parser.add_argument('--checkpoint', help='the path of checkpoint file (eval, serve required)')
    parser.add_argument('--do_test', help='do test while training or not (train required)', action='store_true')
    parser.add_argument('--open_socket', help='open socket server or not (serve required)', action='store_true')

    args = parser.parse_args()

    config_file_path = args.config

    config = configparser.ConfigParser()
    config.read(config_file_path)

    gpu_list = []

    if args.gpu is not None:
        device_list = args.gpu.split(',')

        for device in range(0, len(device_list)):
            gpu_list.append(int(device))

    is_cuda = torch.cuda.is_available()

    information = 'CUDA available: %s' % str(is_cuda)
    logger.info(information)

    if not is_cuda and len(gpu_list) > 0:
        information = 'CUDA is not available but gpu_list is not empty.'
        logger.error(information)
        raise NotImplementedError

    mode = args.mode

    parameters = init_all(config, gpu_list, args.checkpoint, mode)

    if mode == 'serve':
        open_socket = args.open_socket

        if open_socket == 'True':
            serve_socket(parameters, config, gpu_list)
        else:
            serve_simple_IO(parameters, config, gpu_list)
    elif mode == 'train':
        do_test = args.do_test
        train(parameters, config, gpu_list, do_test)
    else:   # mode == 'eval'        
        eval(parameters, config, gpu_list)