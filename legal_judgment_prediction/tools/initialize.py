import logging
import torch

from legal_judgment_prediction.tools.dataset import init_dataset
from legal_judgment_prediction.tools.model import init_model, init_optimizer
from legal_judgment_prediction.tools.output import init_output_function


logger = logging.getLogger(__name__)


def init_all(config, gpu_list, checkpoint, mode, *args, **kwargs):
    result = {}

    information = 'Begin to initialize model.'
    logger.info(information)

    model = init_model(config.get('model', 'model_name'))(config, gpu_list, *args, **kwargs).cuda()

    try:
        model.init_multi_gpu(gpu_list, config, *args, **kwargs)
    except Exception:
        warning = 'No init_multi_gpu implemented in the model, use single gpu instead.'
        logger.warning(warning)

    information = 'Begin to initialize dataset.'
    logger.info(information)

    if mode == 'train':
        result['train_dataset'] = init_dataset(config, task='train', mode='train', *args, **kwargs)
        result['valid_dataset'] = init_dataset(config, task='valid', mode='eval', *args, **kwargs)

        trained_epoch = -1
        optimizer = init_optimizer(model, config, *args, **kwargs)
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
                warning = 'Optimizer has been changed. Use new optimizer to train model.'
                logger.warning(warning)

            if 'global_step' in parameters:
                global_step = parameters['global_step']
    except Exception:
        error = 'Can not load checkpoint file with error %s' % str(Exception)

        if mode != 'train':
            logger.error(error)
            raise Exception
        else:   # mode == 'train'
            logger.warning(error)

    result['model'] = model

    if mode == 'train':
        result['trained_epoch'] = trained_epoch
        result['optimizer'] = optimizer
        result['global_step'] = global_step

    if mode == 'serve':
        result['web_server_IP'] = config.get('server', 'web_server_IP')
        result['web_server_port'] = config.getint('server', 'web_server_port')

        result['server_socket_IP'] = config.get('server', 'server_socket_IP')
        result['server_socket_port'] = config.getint('server', 'server_socket_port')

        result['LINE_CHANNEL_ACCESS_TOKEN'] = config.get('server', 'LINE_CHANNEL_ACCESS_TOKEN')
        result['CHANNEL_SECRET'] = config.get('server', 'CHANNEL_SECRET')
        result['rich_menu_ID'] = config.get('server', 'rich_menu_ID')

    result['output_function'] = init_output_function(config)

    information = 'Initialize done.'
    logger.info(information)

    return result