import logging
import torch.optim as optim

from pytorch_pretrained_bert import BertAdam

from legal_judgment_prediction.tools.model.Bart import LJPBart
from legal_judgment_prediction.tools.model.Bert import LJPBert


logger = logging.getLogger(__name__)


def initialize_model(model_name, *args, **kwargs):
    model_list = {
        'LJPBart': LJPBart,
        'LJPBert': LJPBert
    }

    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        # logger.error('There is no model called %s.' % model_name)
        logger.error(f'There is no model called {model_name}.')
        raise Exception(f'There is no model called {model_name}.')


def initialize_optimizer(config, model, *args, **kwargs):
    optimizer_type = config.get('train', 'optimizer')
    learning_rate = config.getfloat('train', 'learning_rate')
    weight_decay = config.getfloat('train', 'weight_decay')

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'bert_adam':
        optimizer = BertAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        # logger.error('There is no optimizer called %s.' % optimizer_type)
        logger.error(f'There is no optimizer called {optimizer_type}.')
        raise Exception(f'There is no optimizer called {optimizer_type}.')

    return optimizer