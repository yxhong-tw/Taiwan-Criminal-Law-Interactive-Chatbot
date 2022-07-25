import logging
import torch

from legal_judgment_prediction.tools.dataset import initialize_dataloader
from legal_judgment_prediction.tools.model import initialize_model, initialize_optimizer
from legal_judgment_prediction.tools.output import initialize_output_function


logger = logging.getLogger(__name__)


def initialize_all(config, gpu_list, mode, batch_size, checkpoint_path, line_channel_access_token, line_channel_secret, server_socket_ip, *args, **kwargs):
    results = {}

    # information = 'Begin to initialize model.'
    logger.info('Begin to initialize model.')

    model_name = config.get('model', 'model_name')
    model = initialize_model(model_name, *args, **kwargs)(config, *args, **kwargs).cuda()

    try:
        model.initialize_multiple_gpus(gpu_list, *args, **kwargs)
    except Exception:
        # warning = 'No initialize_multiple_gpus implemented in the model, use single gpu instead.'
        logger.warning('No initialize_multiple_gpus implemented in the model, use single gpu instead.')

    # information = 'Begin to initialize dataset.'
    logger.info('Begin to initialize dataset.')

    if batch_size is not None:
        batch_size = int(batch_size)

    if mode == 'train':
        results['train_dataset'] = initialize_dataloader(config, task='train', mode='train', batch_size=batch_size, *args, **kwargs)
        results['valid_dataset'] = initialize_dataloader(config, task='valid', mode='eval', batch_size=batch_size, *args, **kwargs)
        results['test_dataset'] = initialize_dataloader(config, task='test', mode='eval', batch_size=batch_size, *args, **kwargs)

        trained_epoch = -1
        optimizer = initialize_optimizer(config, model, *args, **kwargs)
        global_step = 0
    elif mode == 'eval':
        results['test_dataset'] = initialize_dataloader(config, task='test', mode='eval', batch_size=batch_size, *args, **kwargs)

    if checkpoint_path is not None:
        # checkpoint_path = config.get('model', 'checkpoint_path')

        try:
            parameters = torch.load(checkpoint_path)
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
            # error = 'Can not load checkpoint file with error %s' % str(Exception)
            message = f'Can not load checkpoint file with error {Exception}.'

            if mode != 'train':
                logger.error(message)
                raise Exception(message)
            else:   # mode == 'train'
                logger.warning(message)

    results['model_name'] = model_name
    results['model'] = model

    if mode == 'train':
        results['trained_epoch'] = trained_epoch
        results['optimizer'] = optimizer
        results['global_step'] = global_step

    if mode == 'serve':
        results['web_server_IP'] = config.get('server', 'web_server_IP')
        results['web_server_port'] = config.getint('server', 'web_server_port')

        # results['server_socket_IP'] = config.get('server', 'server_socket_IP')
        results['server_socket_IP'] = server_socket_ip
        results['server_socket_port'] = config.getint('server', 'server_socket_port')

        # results['LINE_CHANNEL_ACCESS_TOKEN'] = config.get('server', 'LINE_CHANNEL_ACCESS_TOKEN')
        # results['CHANNEL_SECRET'] = config.get('server', 'CHANNEL_SECRET')
        results['LINE_CHANNEL_ACCESS_TOKEN'] = line_channel_access_token
        results['CHANNEL_SECRET'] = line_channel_secret
        results['rich_menu_ID'] = config.get('server', 'rich_menu_ID')

    results['output_function'] = initialize_output_function(config)

    information = 'Initialize done.'
    logger.info(information)

    return results