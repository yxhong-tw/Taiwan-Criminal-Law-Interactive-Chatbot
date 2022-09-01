import logging
import torch

from torch.optim import lr_scheduler

from legal_judgment_summarization.model import \
    initialize_model, initialize_optimizer
from legal_judgment_summarization.dataset import initialize_dataloader
from legal_judgment_summarization.output import initialize_output_function
from legal_judgment_summarization.formatter import initialize_formatter


logger = logging.getLogger(__name__)


def initialize_all(
        config
        , mode
        , gpu
        , batch_size
        , checkpoint_path
        , *args
        , **kwargs):
    logger.info('Start to initialize.')

    modes = ['train', 'serve']

    if mode not in modes:
        logger.error(f'There is no mode called {mode}.')
        raise Exception(f'There is no mode called {mode}.')

    # Check gpus and CUDA are available or not.
    # -----
    gpu_list = []

    if gpu == None:
        logger.error('There is no any given gpu.')
        raise Exception('There is no any given gpu.')

    device_list = gpu.replace(' ', '').split(',')

    for device in device_list:
        gpu_list.append(int(device))

    cuda_available = torch.cuda.is_available()

    logger.info(f'CUDA available: {str(cuda_available)}')

    if not cuda_available and len(gpu_list) > 0:
        logger.error('CUDA is not available but gpu_list is not empty.')
        raise Exception('CUDA is not available but gpu_list is not empty.')
    # -----

    model_name = config.get('model', 'model_name')

    model = \
        initialize_model(model_name)(config).cuda()

    try:
        logger.info('Start to initialize multiple gpus.')

        model.initialize_multiple_gpus(gpu_list)

        logger.info('Initialize multiple gpus successfully.')
    except:
        logger.warning('Failed to initialize multiple gpus.')

    results = {
        'gpu_list': gpu_list
    }

    if mode == 'train':
        if batch_size == None:
            logger.warn('There is no batch_size.')
            batch_size = 1
        else:
            batch_size = int(batch_size)

        optimizer = initialize_optimizer(config, model)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(
            optimizer=optimizer
            , milestones=[2, 4, 6]
            , gamma=config.getfloat('train', 'lr_multiplier'))
        trained_epoch = -1
        # global_step = 0

        if checkpoint_path != None:
            parameters = torch.load(checkpoint_path)
            model.load_state_dict(parameters['model'])

            if config.get('train', 'optimizer') == parameters['optimizer_name']:
                optimizer.load_state_dict(parameters['optimizer'])
            else:
                logger.warning('Optimizer has been changed. \
Use new optimizer to train model.')

            exp_lr_scheduler.load_state_dict(parameters['exp_lr_scheduler'])

            trained_epoch = parameters['trained_epoch']

            # if 'global_step' in parameters:
            #     global_step = parameters['global_step']

        train_dataloader = initialize_dataloader(
            config
            , task='train'
            , batch_size=batch_size)
        valid_dataloader = initialize_dataloader(
            config
            , task='valid'
            , batch_size=batch_size)

        results['model'] = model.train()
        results['optimizer_name'] = config.get('train', 'optimizer')
        results['optimizer'] = optimizer
        results['exp_lr_scheduler'] = exp_lr_scheduler
        results['trained_epoch'] = trained_epoch
        # results['global_step'] = global_step
        results['train_dataloader'] = train_dataloader
        results['valid_dataloader'] = valid_dataloader
        results['output_function'] = initialize_output_function(config)
        results['epoch'] = config.getint('train', 'epoch')
        # results['step_size'] = config.getint('train', 'step_size')
        # results['lr_multiplier'] = config.getfloat('train', 'lr_multiplier')
        results['output_path'] = config.get('output', 'output_path')
        results['output_time'] = config.getint('output', 'output_time')
        results['test_time'] = config.getint('output', 'test_time')
    elif mode == 'serve':
        if checkpoint_path == None:
            logger.error(
                'The path of checkpoint is none but the mode is serve.')
            raise Exception(
                'The path of checkpoint is none but the mode is serve.')
    
        parameters = torch.load(checkpoint_path)
        model.load_state_dict(parameters['model'])

        results['model'] = model.eval()
        results['formatter'] = initialize_formatter(config)

    logger.info('Initialize successfully.')

    return results