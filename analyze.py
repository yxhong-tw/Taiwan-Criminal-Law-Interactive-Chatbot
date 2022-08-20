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
    logger = set_logger(log_name=log_name)
    logger.info(information)

    parameters = initialize_all(config=config)

    analyze(parameters=parameters, logger=logger)


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


def analyze(parameters, logger, *args, **kwargs):
    logger.info('Start to analyze data.')

    results = files_analyze(parameters=parameters)
    general_analyze(parameters=parameters, results=results)
    write_back_results(parameters=parameters, results=results)

    logger.info('Analyze data successfully.')


if __name__ == '__main__':
    main()