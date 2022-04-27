import logging
import torch.optim as optim

from pytorch_pretrained_bert import BertAdam

from legal_judgment_prediction.tools.model.Bert import LJPBert


logger = logging.getLogger(__name__)


def init_model(model_name):
    model_list = {
        'LJPBert': LJPBert
    }

    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        logger.error('There is no model called %s.' % model_name)
        raise AttributeError


def init_optimizer(model, config, *args, **kwargs):
    optimizer_type = config.get('train', 'optimizer')
    learning_rate = config.getfloat('train', 'learning_rate')

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=config.getfloat('train', 'weight_decay'))
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=config.getfloat('train', 'weight_decay'))
    elif optimizer_type == 'bert_adam':
        optimizer = BertAdam(model.parameters(), lr=learning_rate, weight_decay=config.getfloat('train', 'weight_decay'))
    else:
        logger.error('There is no optimizer called %s.' % optimizer_type)
        raise AttributeError

    return optimizer