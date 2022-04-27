from asyncio.log import logger
import logging
from tools.formatter.Bert import BertLJP


logger = logging.getLogger(__name__)


def init_formatter(config, task, *args, **kwargs):
    formatter = choose_formatter(config, task, *args, **kwargs)

    def collate_fn(data):
        return formatter.process(data, config, task)

    return collate_fn


def choose_formatter(config, task, *args, **kwargs):
    formatter_list = {
        'BertLJP': BertLJP
    }

    try:
        formatter_type = config.get('data', '%s_formatter_type' % task)
    except Exception:
        logger.error('%s_formatter_type has not been defined in config file.' % task)
        raise AttributeError

    if formatter_type in formatter_list:
        formatter = formatter_list[formatter_type](config, task, *args, **kwargs)
        return formatter
    else:
        logger.error('There is no formatter called %s.' % formatter_type)
        raise AttributeError