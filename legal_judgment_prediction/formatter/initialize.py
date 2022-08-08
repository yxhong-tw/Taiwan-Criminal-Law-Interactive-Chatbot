import logging

from legal_judgment_prediction.formatter.bart import BartFormatter
from legal_judgment_prediction.formatter.bert import BertFormatter


logger = logging.getLogger(__name__)


def initialize_formatter(config, mode, task=None, *args, **kwargs):
    logger.info('Start to initialize formatter.')

    formatter = choose_formatter(config, mode, task)

    def collate_fn(data):
        return formatter.process(data)

    logger.info('Initialize formatter successfully.')

    return collate_fn


def choose_formatter(config, mode, task=None, *args, **kwargs):
    formatter_types = {
        'BartFormatter': BartFormatter,
        'BertFormatter': BertFormatter
    }

    try:
        formatter_type = config.get('data', f'{task}_formatter_type')
    except Exception:
        if config.get('model', 'model_name') == 'LJPBart':
            formatter_type = 'BartFormatter'
        elif config.get('model', 'model_name') == 'LJPBert':
            formatter_type = 'BertFormatter'
        else:
            logger.error('Failed to get the type of formatter.')
            raise Exception

    if formatter_type in formatter_types:
        formatter = \
            formatter_types[formatter_type](config, mode)

        return formatter
    else:
        logger.error(f'There is no formatter called {formatter_type}.')
        raise Exception(f'There is no formatter called {formatter_type}.')