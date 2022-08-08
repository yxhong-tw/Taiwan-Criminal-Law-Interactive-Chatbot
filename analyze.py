import sys
import argparse
import configparser
import logging

from data_analysis.initialize import initialize_all
from data_analysis.utils import \
    files_analyze, general_analyze, write_back_results


information = ' '.join(sys.argv)


def main(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config'
        , help='The path of config file'
        , required=True
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

    parameters = initialize_all(config)

    analyze(parameters, logger)


def analyze(parameters, logger):
    logger.info('Start to analyze data.')

    results = files_analyze(parameters)
    general_analyze(parameters, results)
    write_back_results(parameters, results)

    logger.info('Analyze data successfully.')


if __name__ == '__main__':
    main()