import sys
import argparse
import configparser
import logging
import torch
import threading

from legal_judgment_prediction.initialize import initialize_all
from legal_judgment_prediction.train import train
from legal_judgment_prediction.eval import eval
from legal_judgment_prediction.serve import serve_socket, serve_simple_IO
from line_bot.app import App_Thread


information = ' '.join(sys.argv)


def main(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config'
        , help='The path of config file'
        , required=True
    )
    parser.add_argument(
        '-m', '--mode'
        , help='train, eval or serve'
        , required=True
    )
    parser.add_argument(
        '-g', '--gpu'
        , help='The list of gpu IDs'
    )
    parser.add_argument(
        '-cp', '--checkpoint_path'
        , help='The path of checkpoint (Ignore if you do not use checkpoint)'
    )
    parser.add_argument(
        '-bs', '--batch_size'
        , help='The batch size in train or eval mode'
    )
    parser.add_argument(
        '-dt', '--do_test'
        , help='Test in train mode (Ignore if you do not test)'
        , action='store_true'
    )
    parser.add_argument(
        '-os', '--open_server'
        , help='Open server in serve mode (Ignore if you do not open server)'
        , action='store_true'
    )
    parser.add_argument(
        '-lcat', '--line_channel_access_token'
        , help='The channel access token of LINE bot'
    )
    parser.add_argument(
        '-lcs', '--line_channel_secret'
        , help='The channel secret of LINE bot'
    )
    parser.add_argument(
        '-ssip', '--server_socket_ip'
        , help='The IP of server socket'
    )

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    log_name = config.get('log', 'name')

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)

    fh = logging.FileHandler(
        filename=f'legal_judgment_prediction/logs/{log_name}'
        , mode='a'
        , encoding='UTF-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    logger.addHandler(fh)
    
    logger.info(information)

    gpu_list = []

    if args.gpu is not None:
        device_list = args.gpu.replace(' ', '').split(',')

        for device in range(0, len(device_list)):
            gpu_list.append(int(device))

    cuda_available = torch.cuda.is_available()

    logger.info(f'CUDA available: {str(cuda_available)}')

    if not cuda_available and len(gpu_list) > 0:
        logger.error('CUDA is not available but gpu_list is not empty.')
        raise Exception('CUDA is not available but gpu_list is not empty.')

    parameters = initialize_all(
        config
        , gpu_list
        , args.mode
        , args.batch_size
        , args.checkpoint_path
        , args.line_channel_access_token
        , args.line_channel_secret
        , args.server_socket_ip)

    if args.mode == 'train':
        train(parameters, config, gpu_list, args.do_test)
    elif args.mode == 'eval':  
        eval(parameters, config, gpu_list)
    elif args.mode == 'serve':
        if args.open_server == True:
            ljp_thread = threading.Thread(
                target=serve_socket, args=(parameters, config))
            ljp_thread.start()

            line_bot_thread = App_Thread(parameters)
            line_bot_thread.start()

            ljp_thread.join()

            line_bot_thread.shutdown()
            line_bot_thread.join()
        else:
            serve_simple_IO(parameters, config)
    else:
        logger.error('Invalid mode, please check again.')
        raise Exception('Invalid mode, please check again.')


if __name__ == '__main__':
    main()