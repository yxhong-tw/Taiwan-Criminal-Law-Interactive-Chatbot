import logging
import os


logger = logging.getLogger(__name__)


def initialize_all(config):
    logger.info('Start to initialize.')

    check_table = {
        'types': ['CAIL2020_sfzy', 'taiwan_indictments', 'CNewSum_v2']
    }

    results = {
        'name': config.get('common', 'name')
        , 'type': config.get('common', 'type')
        , 'data_path': config.get('common', 'data_path')
        , 'output_path': config.get('common', 'output_path')
    }

    if results['type'] not in check_table['types']:
        logger.error(f'There is no name called {results["type"]}.')
        raise Exception(f'There is no name called {results["type"]}.')

    if not os.path.exists(results['data_path']):
        logger.error(f'The path of data {results["data_path"]} does not exist.')
        raise Exception(
            f'The path of data {results["data_path"]} does not exist.')

    if not os.path.exists(results['output_path']):
        logger.warn(
            f'The path of output {results["output_path"]} does not exist.')
        logger.info('Make the directory automatically.')

        os.makedirs(results['output_path'])

    logger.info('Initialize successfully.')

    return results