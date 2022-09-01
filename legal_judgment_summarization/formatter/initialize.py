import logging

from legal_judgment_summarization.formatter.bart import BartFormatter


logger = logging.getLogger(__name__)


def initialize_formatter(config, task=None, *args, **kwargs):
    logger.info('Start to initialize formatter.')

    formatter = choose_formatter(config=config, task=task)

    def collate_fn(data):
        return formatter.process(data)

    logger.info('Initialize formatter successfully.')

    return collate_fn


def choose_formatter(config, task=None, *args, **kwargs):
    formatter_types = {'BartFormatter': BartFormatter}

    try:
        formatter_type = config.get('data', f'{task}_formatter_type')
    except:
        formatter_type = 'BartFormatter'

    if formatter_type in formatter_types:
        formatter = \
            formatter_types[formatter_type](config)

        return formatter
    else:
        logger.error(f'There is no formatter called {formatter_type}.')
        raise Exception(f'There is no formatter called {formatter_type}.')