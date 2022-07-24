import sys
import argparse
import configparser
import logging

from legal_judgment_prediction.tools.generate.initialize import initialize_all
from legal_judgment_prediction.tools.generate.utils import top_50_article, all


information = ' '.join(sys.argv)

def main(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='The path of config file', required=True)
    parser.add_argument('--data_type', '-dt', help='one_label or multi_labels', required=True)
    parser.add_argument('--mode', '-m', help='top_50_article or all', required=True)

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    log_name = config.get('log', 'name')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)

    fh = logging.FileHandler(f'legal_judgment_prediction/logs/{log_name}', mode='a', encoding='UTF-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    logger.addHandler(fh)
    
    logger.info(information)

    parameters = initialize_all(config, args.data_type, args.mode)

    if args.mode == 'top_50_article':
        top_50_article(parameters)
    elif args.mode == 'all':
        all(parameters)


if __name__ == '__main__':
    main()