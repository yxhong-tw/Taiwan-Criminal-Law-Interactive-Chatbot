import logging

from legal_judgment_prediction.tools.formatter.Bart import BartLJP
from legal_judgment_prediction.tools.formatter.Bert import BertLJP


logger = logging.getLogger(__name__)


def initialize_formatter(config, task, mode, *args, **kwargs):
    formatter = choose_formatter(config, task, mode, *args, **kwargs)

    def collate_fn(data):
        return formatter.process(data)

    return collate_fn


def choose_formatter(config, task, mode, *args, **kwargs):
    formatter_list = {
        'BartLJP': BartLJP,
        'BertLJP': BertLJP
    }

    # formatter_type = config.get('data', '%s_formatter_type' % task)
    formatter_type = config.get('data', f'{task}_formatter_type')

    if formatter_type in formatter_list:
        formatter = formatter_list[formatter_type](config, mode, *args, **kwargs)
        return formatter
    else:
        # logger.error('There is no formatter called %s.' % formatter_type)
        logger.error(f'There is no formatter called {formatter_type}.')
        raise Exception(f'There is no formatter called {formatter_type}.')