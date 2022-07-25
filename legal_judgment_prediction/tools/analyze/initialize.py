import logging


logger = logging.getLogger(__name__)


def initialize_all(config):
    logger.info('Start to initialize.')

    results = {}

    results['name'] = config.get('common', 'name')
    results['data_path'] = config.get('common', 'data_path')
    results['output_path'] = config.get('common', 'output_path')

    logger.info('Initialize successfully.')

    return results