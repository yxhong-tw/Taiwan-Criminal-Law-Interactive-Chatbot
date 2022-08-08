import logging
import torch

from legal_judgment_prediction.dataset import initialize_dataloader
from legal_judgment_prediction.model import \
    initialize_model, initialize_optimizer
from legal_judgment_prediction.formatter import initialize_formatter
from legal_judgment_prediction.output import initialize_output_function


logger = logging.getLogger(__name__)


def initialize_all(
        config
        , gpu_list
        , mode
        , batch_size
        , checkpoint_path
        , line_channel_access_token
        , line_channel_secret
        , server_socket_ip
        , *args
        , **kwargs):
    logger.info('Start to initialize.')

    model_name = config.get('model', 'model_name')

    results = {
        'model_name': model_name
        , 'formatter': initialize_formatter(config, mode)
        , 'output_function': initialize_output_function(config)
    }

    model = \
        initialize_model(model_name)(config)
    model = model.cuda()

    try:
        logger.info('Start to initialize multiple gpus.')

        model.initialize_multiple_gpus(gpu_list)

        logger.info('Initialize multiple gpus successfully.')
    except:
        logger.warning('Failed to initialize multiple gpus.')

    if checkpoint_path is not None:
        parameters = torch.load(checkpoint_path)
        model.load_state_dict(parameters['model'])

        if mode == 'train':
            trained_epoch = parameters['trained_epoch']

            if config.get('train', 'optimizer') == parameters['optimizer_name']:
                optimizer.load_state_dict(parameters['optimizer'])
            else:
                logger.warning('Optimizer has been changed. \
Use new optimizer to train model.')

            if 'global_step' in parameters:
                global_step = parameters['global_step']

    results['model'] = model

    if batch_size is None and (mode == 'train' or mode == 'eval'):
        logger.warn(f'There is no batch_size but mode is \'{mode}\'.')
        batch_size = 1
    else:
        batch_size = int(batch_size)

    if mode == 'train':
        train_dataloader = initialize_dataloader(
            config
            , task='train'
            , mode='train'
            , batch_size=batch_size)
        valid_dataloader = initialize_dataloader(
            config
            , task='valid'
            , mode='eval'
            , batch_size=batch_size)
        test_dataloader = initialize_dataloader(
            config
            , task='test'
            , mode='eval'
            , batch_size=batch_size)

        trained_epoch = -1
        optimizer = initialize_optimizer(config, model)
        global_step = 0
    elif mode == 'eval':
        test_dataloader = initialize_dataloader(
            config
            , task='test'
            , mode='eval'
            , batch_size=batch_size)

    if mode == 'train':
        results['model'].train()
        results['train_dataloader'] = train_dataloader
        results['valid_dataloader'] = valid_dataloader
        results['test_dataloader'] = test_dataloader
        results['trained_epoch'] = trained_epoch
        results['optimizer'] = optimizer
        results['global_step'] = global_step
    elif mode == 'eval':
        results['model'].eval()
        results['test_dataloader'] = test_dataloader
    elif mode == 'serve':
        results['model'].eval()
        results['web_server_IP'] = config.get('server', 'web_server_IP')
        results['web_server_port'] = config.getint('server', 'web_server_port')
        results['server_socket_IP'] = server_socket_ip
        results['server_socket_port'] = \
            config.getint('server', 'server_socket_port')
        results['LINE_CHANNEL_ACCESS_TOKEN'] = line_channel_access_token
        results['CHANNEL_SECRET'] = line_channel_secret
        results['rich_menu_ID'] = config.get('server', 'rich_menu_ID')

    logger.info('Initialize successfully.')

    return results