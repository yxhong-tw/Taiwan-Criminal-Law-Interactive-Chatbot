import sys
import logging
import argparse

from chinese_conversion.initialize import initialize_all
from chinese_conversion.utils import chinese_conversion


information = ' '.join(sys.argv)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)

fh = logging.FileHandler(
    filename='logs/chinese_conversion.log'
    , mode='a'
    , encoding='UTF-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(sh)
logger.addHandler(fh)

logger.info(information)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sdp', '--source_directory_path'
        , help='The path of source file'
        , required=True
    )
    parser.add_argument(
        '-ddp', '--destination_directory_path'
        , help='The path of destination directory'
        , required=True
    )
    parser.add_argument(
        '-c', '--config'
        , help='The config of OpenCC'
    )

    args = parser.parse_args()

    parameters = initialize_all(args)

    chinese_conversion(parameters)


if __name__ == '__main__':
    main()