import sys
import argparse
import configparser
import logging

from legal_judgment_summarization.initialize import initialize_all
from legal_judgment_summarization.train import train
from legal_judgment_summarization.serve import serve


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
        , help='train or serve'
        , required=True
    )
    parser.add_argument(
        '-g', '--gpu'
        , help='The list of gpu IDs'
        , required=True
    )
    parser.add_argument(
        '-cp', '--checkpoint_path'
        , help='The path of checkpoint (Ignore if you do not use checkpoint)'
    )
    parser.add_argument(
        '-bs', '--batch_size'
        , help='The batch size in train and eval mode'
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
        filename=f'logs/{log_name}'
        , mode='a'
        , encoding='UTF-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    logger.addHandler(fh)
    
    logger.info(information)

    parameters = initialize_all(
        config
        , args.mode
        , args.gpu
        , args.batch_size
        , args.checkpoint_path)

    if args.mode == 'train':
        train(parameters)
    elif args.mode == 'serve':
        serve(parameters)


if __name__ == '__main__':
    main()