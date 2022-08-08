import logging
import torch.optim as optim

from pytorch_pretrained_bert import BertAdam

from legal_judgment_prediction.model.bart import LJPBart
from legal_judgment_prediction.model.bert import LJPBert


logger = logging.getLogger(__name__)


def initialize_model(model_name, *args, **kwargs):
    logger.info('Start to initialize model.')

    model_types = {
        'LJPBart': LJPBart,
        'LJPBert': LJPBert
    }

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

    optimizer_types = {
        'adam': optim.Adam
        , 'sgd': optim.SGD
        , 'bert_adam': BertAdam
    }

    if optimizer_name in optimizer_types:
        logger.info('Initialize optimizer successfully.')

        return optimizer_types[optimizer_name](
            model.parameters()
            , lr=learning_rate
            , weight_decay=weight_decay)
    else:
        logger.error(f'There is no optimizer called {optimizer_name}.')
        raise Exception(f'There is no optimizer called {optimizer_name}.')