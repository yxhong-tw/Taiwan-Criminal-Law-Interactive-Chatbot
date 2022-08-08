import sys
import argparse
import configparser
import logging
import torch

from data_generation.initialize import initialize_all
from data_generation.utils import \
    get_summarization_data, top_50_articles, all \
    , get_innocence_data, combine_data


information = ' '.join(sys.argv)


def main(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config'
        , help='The path of config file'
        , required=True
    )
    parser.add_argument(
        '-g', '--gpu'
        , help='The list of gpu IDs (This is required if mode is not analyze.)'
    )
    parser.add_argument(
        '-cp', '--checkpoint_path'
        , help='The path of checkpoint (Ignore if you do not use checkpoint)'
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

    parameters = initialize_all(config, gpu_list, args.checkpoint_path)

    generate(parameters, logger)


def generate(parameters, logger):
    if parameters['type'] == 'summarization':
        logger.info(f'Start to generate {parameters["type"]} T.V. dataset.')

        get_summarization_data(parameters)

        logger.info(f'Generate {parameters["type"]} T.V. dataset successfully.')
    elif parameters['type'] == 'crime':
        logger.info(f'Start to generate {parameters["type"]} T.V.T. dataset \
in {parameters["label"]} label and {parameters["range"]} range.')

        if parameters['range'] == 'top_50_articles':
            top_50_articles(parameters)
        elif parameters['range'] == 'all':
            all(parameters)

        logger.info(f'Generate {parameters["type"]} T.V.T. dataset \
in {parameters["label"]} label and {parameters["range"]} range successfully.')
    elif parameters['type'] == 'innocence':
        logger.info(f'Start to generate {parameters["type"]} T.V.T. dataset.')

        get_innocence_data(parameters)

        logger.info(
            f'Generate {parameters["type"]} T.V.T. dataset successfully.')
    elif parameters['type'] == 'combination':
        logger.info('Start to generate combination dataset.')

        combine_data(parameters)

        logger.info('Generate combination dataset successfully.')


if __name__ == '__main__':
    main()