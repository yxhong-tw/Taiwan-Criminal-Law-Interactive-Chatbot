import sys
import argparse
import configparser
import logging

from data_generation.initialize import initialize_all
from data_generation.utils import \
    get_common_data, convert_fact_to_summarization, get_summarization_data


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
    logger = set_logger(log_name=log_name)
    logger.info(information)

    parameters = initialize_all(config, args.gpu, args.checkpoint_path)

    generate(parameters, logger)


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


def generate(parameters, logger):
    logger.info(f'Start to generate data.')

    if parameters['task'] == 'legal_judgment_prediction':
        if parameters['summarization'] == 'before':
            get_common_data(parameters=parameters)
        else:
            convert_fact_to_summarization(parameters)
    elif parameters['task'] == 'text_summarization':
        get_summarization_data(parameters)

    logger.info('Generate data successfully.')


if __name__ == '__main__':
    main()