import logging
import os
import torch

from torch.optim import lr_scheduler

from legal_judgment_prediction.model import \
    initialize_model, initialize_optimizer
from legal_judgment_prediction.dataset import initialize_dataloader
from legal_judgment_prediction.output import initialize_output_function
from legal_judgment_prediction.formatter import initialize_formatter
from utils import get_tables


logger = logging.getLogger(__name__)


def initialize_all(
        config
        , mode
        , device_str
        , checkpoint_path
        , batch_size
        , do_test
        , line_channel_access_token
        , line_channel_secret
        , server_socket_ip
        , *args
        , **kwargs):
    logger.info('Start to initialize.')

    check_mode(mode=mode)

    gpus = check_and_set_gpus(device_str=device_str)
    model = initialize_model(config=config)

    try:
        logger.info('Start to initialize multiple gpus.')

        model.initialize_multiple_gpus(gpus=gpus)

        logger.info('Initialize multiple gpus successfully.')
    except:
        logger.warning('Failed to initialize multiple gpus.')

    results = {}

    if mode == 'train':
        optimizer = initialize_optimizer(config=config, model=model)

        milestones_str_list = config.get('train', 'milestones').split(',')
        milestones = [int(milestone) for milestone in milestones_str_list]
        gamma = config.getfloat('train', 'lr_multiplier')

        exp_lr_scheduler = lr_scheduler.MultiStepLR(
            optimizer=optimizer
            , milestones=milestones
            , gamma=gamma)

        optimizer_name = config.get('train', 'optimizer')
        trained_epoch = -1

        if checkpoint_path != None:
            if not os.path.exists(path=checkpoint_path):
                logger.error(
                    'The path of checkpoint is not none but it does not exixt.')
                raise Exception(
                    'The path of checkpoint is not none but it does not exixt.')

            parameters = torch.load(f=checkpoint_path)
            model.load_state_dict(parameters['model'])

            if optimizer_name == parameters['optimizer_name']:
                optimizer.load_state_dict(parameters['optimizer'])
            else:
                logger.warning('Optimizer has been changed.')
                logger.info('Use the new optimizer to train the model.')

            exp_lr_scheduler.load_state_dict(parameters['exp_lr_scheduler'])

            trained_epoch = parameters['trained_epoch']
        else:
            logger.warn('The path of checkpoint is none.')

        batch_size = check_and_set_batch_size(batch_size=batch_size)

        train_dataloader = initialize_dataloader(
            config=config
            , task='train'
            , mode='train'
            , batch_size=batch_size)
        valid_dataloader = initialize_dataloader(
            config=config
            , task='valid'
            , mode='eval'
            , batch_size=batch_size)

        if do_test == True:
            test_dataloader = initialize_dataloader(
                config=config
                , task='test'
                , mode='eval'
                , batch_size=batch_size)

            results['test_dataloader'] = test_dataloader

        output_function = initialize_output_function(config=config)

        output_path = config.get('output', 'output_path')

        if not os.path.exists(path=output_path):
            logger.warn(
                f'The path of output {output_path} does not exist.')
            logger.info('Make the directory automatically.')

            os.makedirs(name=output_path)

        results['model'] = model.cuda().train()
        results['optimizer'] = optimizer
        results['exp_lr_scheduler'] = exp_lr_scheduler
        results['optimizer_name'] = config.get('train', 'optimizer')
        results['trained_epoch'] = trained_epoch
        results['train_dataloader'] = train_dataloader
        results['valid_dataloader'] = valid_dataloader
        results['output_function'] = output_function
        results['output_path'] = output_path
        results['model_name'] = config.get('model', 'model_name')
        results['total_epoch'] = config.getint('train', 'total_epoch')
        results['output_time'] = config.getint('output', 'output_time')
        results['test_time'] = config.getint('output', 'test_time')
    elif mode == 'eval':
        # if checkpoint_path == None:
        #     logger.error('The path of checkpoint is none.')
        #     raise Exception('The path of checkpoint is none.')

        # parameters = torch.load(f=checkpoint_path)
        # model.load_state_dict(parameters['model'])

        batch_size = check_and_set_batch_size(batch_size=batch_size)

        test_dataloader = initialize_dataloader(
            config=config
            , task='test'
            , mode='eval'
            , batch_size=batch_size)

        output_function = initialize_output_function(config=config)

        results['model'] = model.cuda().eval()
        results['test_dataloader'] = test_dataloader
        results['output_function'] = output_function
        results['model_path'] = config.get('model', 'model_path')
        results['output_function_name'] = \
            config.get('output', 'output_function')
    elif mode == 'serve':
        if checkpoint_path == None:
            logger.error('The path of checkpoint is none.')
            raise Exception('The path of checkpoint is none.')

        parameters = torch.load(f=checkpoint_path)
        model.load_state_dict(parameters['model'])

        formatter = initialize_formatter(config=config, mode=mode)

        model_name = config.get('model', 'model_name')

        if model_name == 'LJPBert':
            articles_table, article_sources_table, accusations_table = \
                get_tables(config=config, formatter=formatter)

        results['LINE_CHANNEL_ACCESS_TOKEN'] = line_channel_access_token
        results['CHANNEL_SECRET'] = line_channel_secret
        results['server_socket_IP'] = server_socket_ip
        results['model'] = model.cuda().eval()
        results['formatter'] = formatter
        results['model_name'] = model_name
        results['articles_table'] = articles_table
        results['article_sources_table'] = article_sources_table
        results['accusations_table'] = accusations_table
        results['web_server_IP'] = config.get('server', 'web_server_IP')
        results['web_server_port'] = config.getint('server', 'web_server_port')
        results['server_socket_port'] = \
            config.getint('server', 'server_socket_port')

    logger.info('Initialize successfully.')

    return results


def check_mode(mode):
    modes = ['train', 'eval', 'serve']

    if mode not in modes:
        logger.error(f'There is no mode called {mode}.')
        raise Exception(f'There is no mode called {mode}.')


def check_and_set_gpus(device_str):
    gpus = []

    if device_str == None:
        logger.error('There is no any given gpu.')
        raise Exception('There is no any given gpu.')
        
    devices = device_str.replace(' ', '').split(',')

    for device in devices:
        gpus.append(int(device))

    cuda_status = torch.cuda.is_available()

    logger.info(f'CUDA available: {cuda_status}')

    if not cuda_status and len(gpus) > 0:
        logger.error('CUDA is not available but gpu_list is not empty.')
        raise Exception('CUDA is not available but gpu_list is not empty.')

    return gpus


def check_and_set_batch_size(batch_size):
    if batch_size == None:
        logger.warn(f'There is no batch_size.')
        logger.info('Set the batch_size to 1 to continue.')
        batch_size = 1
    else:
        batch_size = int(batch_size)

    return batch_size