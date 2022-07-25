import logging

from legal_judgment_prediction.tools.generate.initialize import initialize_all
from legal_judgment_prediction.tools.generate.utils import top_50_article, all


logger = logging.getLogger(__name__)


def generate(config):
    parameters = initialize_all(config)

    logger.info(f'Start to generate {parameters["label"]} T.V.T. dataset in {parameters["range"]} range.')

    if parameters['range'] == 'top_50_article':
        top_50_article(parameters)
    elif parameters['range'] == 'all':
        all(parameters)
    else:
        raise Exception('There is no range named' + ' ' + parameters['range'] + '.')

    logger.info(f'Generate {parameters["label"]} T.V.T. dataset in {parameters["range"]} range successfully.')
