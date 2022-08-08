import logging
import torch.optim as optim

from legal_judgment_summarization.model.bart import LJSBart


logger = logging.getLogger(__name__)


def initialize_model(model_name, *args, **kwargs):
    logger.info('Start to initialize model.')

    model_types = {'LJSBart': LJSBart}

    if model_name in model_types.keys():
        logger.info('Initialize model successfully.')

        return model_types[model_name]
    else:
        logger.error(f'There is no model called {model_name}.')
        raise Exception(f'There is no model called {model_name}.')


def initialize_optimizer(config, model, *args, **kwargs):
    logger.info('Start to initialize optimizer.')

    optimizer_name = config.get('train', 'optimizer')
    learning_rate = config.getfloat('train', 'learning_rate')
    weight_decay = config.getfloat('train', 'weight_decay')

    optimizer_types = {'adam': optim.Adam}

    if optimizer_name in optimizer_types:
        logger.info('Initialize optimizer successfully.')

        return optimizer_types[optimizer_name](
            params=model.parameters()
            , lr=learning_rate
            , weight_decay=weight_decay)
    else:
        logger.error(f'There is no optimizer called {optimizer_name}.')
        raise Exception(f'There is no optimizer called {optimizer_name}.')