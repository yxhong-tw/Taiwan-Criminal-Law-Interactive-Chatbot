import sys
import argparse
import configparser
import logging
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

    logger = set_logger(log_name=config.get('log', 'name'))
    logger.info(information)

    parameters = initialize_all(
        config=config
        , mode=args.mode
        , device_str=args.gpu
        , checkpoint_path=args.checkpoint_path
        , batch_size=args.batch_size
        , do_test=args.do_test
        , line_channel_access_token=args.line_channel_access_token
        , line_channel_secret=args.line_channel_secret
        , server_socket_ip=args.server_socket_ip)

    if args.mode == 'train':
        train(parameters, args.do_test)
    elif args.mode == 'eval':  
        eval(parameters)
    elif args.mode == 'serve':
        if args.open_server == True:
            ljp_thread = threading.Thread(
                target=serve_socket, args=(parameters))
            ljp_thread.start()

            line_bot_thread = App_Thread(parameters=parameters)
            line_bot_thread.start()

            ljp_thread.join()

            line_bot_thread.shutdown()
            line_bot_thread.join()
        else:
            serve_simple_IO(parameters=parameters)


def set_logger(log_name, *args, **kwargs):
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)

    fh = logging.FileHandler(
        filename=f'logs/{log_name}'
        , mode='a'
        , encoding='UTF-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger


if __name__ == '__main__':
    main()