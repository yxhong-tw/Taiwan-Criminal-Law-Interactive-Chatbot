import logging
import sys
import argparse
import configparser
import torch
import threading

from legal_judgment_prediction.tools.initialize import initialize_all
from legal_judgment_prediction.tools.train import train
from legal_judgment_prediction.tools.eval import eval
from legal_judgment_prediction.tools.serve.serve import serve_socket, serve_simple_IO
from line_bot.app import App_Thread


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)

# bart.log or bert.log
fh = logging.FileHandler('legal_judgment_prediction/log/bart.log', mode='a', encoding='UTF-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(sh)
logger.addHandler(fh)

information = ' '.join(sys.argv)
logger.info(information)


def main(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='The path of config file', required=True)
    parser.add_argument('--gpu', '-g', help='The list of gpu IDs', required=True)
    parser.add_argument('--mode', help='Train, eval or serve', required=True)
    # parser.add_argument('--checkpoint', help='the path of checkpoint file (eval, serve required)')
    parser.add_argument('--use_checkpoint', help='Use checkpoint (Ignore if do not use checkpoint)')
    parser.add_argument('--do_test', help='Do test while training (Ignore if do not test while training)')
    parser.add_argument('--open_server', help='Open web server while serving (Ignore if do not open web server while serving')

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    gpu_list = []
    if args.gpu is not None:
        device_list = args.gpu.replace(' ', '').split(',')

        for device in range(0, len(device_list)):
            gpu_list.append(int(device))

    cuda_available = torch.cuda.is_available()

    # information = 'CUDA available: %s' % str(cuda_available)
    logger.info(f'CUDA available: {str(cuda_available)}')

    if not cuda_available and len(gpu_list) > 0:
        # information = 'CUDA is not available but gpu_list is not empty.'
        logger.error('CUDA is not available but gpu_list is not empty.')
        raise Exception('CUDA is not available but gpu_list is not empty.')

    parameters = initialize_all(config, gpu_list, args.mode, args.use_checkpoint)

    if args.mode == 'train':
        train(parameters, config, gpu_list, args.do_test)
    elif args.mode == 'eval':  
        eval(parameters, config, gpu_list)
    elif args.mode == 'serve':
        if args.open_server == True:
            ljp_thread = threading.Thread(target=serve_socket, args=(parameters, config, gpu_list))
            ljp_thread.start()

            line_bot_thread = App_Thread(parameters)
            line_bot_thread.start()

            ljp_thread.join()

            line_bot_thread.shutdown()
            line_bot_thread.join()
        else:
            serve_simple_IO(parameters, config, gpu_list)
    else:
        logger.error('Invalid mode, please check again.')
        raise Exception('Invalid mode, please check again.')


if __name__ == '__main__':
    main()