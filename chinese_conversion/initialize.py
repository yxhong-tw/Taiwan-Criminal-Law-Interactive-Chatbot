import logging
import os
import opencc


logger = logging.getLogger(__name__)


def initialize_all(args):
    logger.info('Start to initialize.')

    if not os.path.exists(args.source_directory_path):
        logger.error('The path of the source directory does not exist.')
        raise Exception

    if not os.path.exists(args.destination_directory_path):
        logger.warning('The path of the destination directory does not exist.')
        logger.info('Make directory automatically.')

        os.mkdir(args.destination_directory_path)

    configs = [
        's2t.json'
        , 't2s.json'
        , 's2tw.json'
        , 'tw2s.json'
        , 's2hk.json'
        , 'hk2s.json'
        , 's2twp.json'
        , 'tw2sp.json'
        , 't2tw.json'
        , 'hk2t.json'
        , 't2hk.json'
        , 't2jp.json'
        , 'jp2t.json'
        , 'tw2t.json'
    ]

    if args.config not in configs:
        logger.error(f'There is no config called {parameters["config"]}.')
        raise Exception

    parameters = {
        'source_directory_path': args.source_directory_path
        , 'destination_directory_path': args.destination_directory_path
        , 'config': args.config
        , 'converter': opencc.OpenCC(args.config)
    }

    logger.info('Initialize successfully.')

    return parameters