import logging
import torch

from legal_judgment_prediction.tools.dataset import init_dataset
from legal_judgment_prediction.tools.model import init_model, init_optimizer
from legal_judgment_prediction.tools.output import init_output_function


logger = logging.getLogger(__name__)


def init_all(config, gpu_list, checkpoint, mode, *args, **kwargs):
    result = {}

    information = 'Begin to initialize dataset and formatter...'
    logger.info(information)

    information = 'Begin to initialize model...'
    logger.info(information)

    model = init_model(config.get('model', 'model_name'))(config, gpu_list, *args, **kwargs).cuda()

    try:
        model.init_multi_gpu(gpu_list, config, *args, **kwargs)
    except Exception:
        information = 'No init_multi_gpu implemented in the model, use single gpu instead.'
        logger.warning(information)

    if mode == 'train':
        result['train_dataset'] = init_dataset(config, task='train', mode='train', *args, **kwargs)
        result['valid_dataset'] = init_dataset(config, task='valid', mode='eval', *args, **kwargs)

        optimizer = init_optimizer(model, config, *args, **kwargs)
        trained_epoch = -1
        global_step = 0
    elif mode == 'eval':
        result['test_dataset'] = init_dataset(config, task='test', mode='eval', *args, **kwargs)

    try:
        parameters = torch.load(checkpoint)
        model.load_state_dict(parameters['model'])

        if mode == 'train':
            trained_epoch = parameters['trained_epoch']

            if config.get('train', 'optimizer') == parameters['optimizer_name']:
                optimizer.load_state_dict(parameters['optimizer'])
            else:
                information = 'Optimizer has been changed. Use new optimizer to train model.'
                logger.warning(information)

            if 'global_step' in parameters:
                global_step = parameters['global_step']

            result['optimizer'] = optimizer
            result['trained_epoch'] = trained_epoch
            result['global_step'] = global_step
    except Exception:
        information = 'Can not load checkpoint file with error %s' % str(Exception)

        if mode != 'train':
            logger.error(information)
            raise Exception
        else:   # mode == 'train'
            logger.warning(information)

    result['model'] = model
    result['output_function'] = init_output_function(config)

    logger.info('Initialize done.')

    return result