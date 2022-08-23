import logging
import pickle
import os
import torch

# from legal_judgment_prediction.model import initialize_model
# from legal_judgment_prediction.formatter import initialize_formatter
from legal_judgment_summarization.model import initialize_model
from legal_judgment_summarization.formatter import initialize_formatter


logger = logging.getLogger(__name__)


def initialize_all(config, device_str, checkpoint_path):
    logger.info('Start to initialize.')

    check_table = {
        'tasks': ['legal_judgment_prediction', 'text_summarization']
        , 'summarizations': ['none', 'bart', 'lead_3']
        , 'model_names': ['LJSBart']
        , 'types': [
            'one_label'
            , 'multi_labels'
            , 'combination'
            , 'CAIL2020_sfzy'
            , 'CNewSum_v2']
    }

    results = {
        'task': config.get('common', 'task')
        , 'type': config.get('common', 'type')
        , 'data_path': config.get('common', 'data_path')
        , 'output_path': config.get('common', 'output_path')
        , 'train_size': config.getfloat('common', 'train_size')
        , 'valid_size': config.getfloat('common', 'valid_size')
        , 'generate_test_data': \
            config.getboolean('common', 'generate_test_data')
        , 'random_seed': config.getint('common', 'random_seed')
    }

    if results['task'] not in check_table['tasks']:
        logger.error(f'There is no task called {results["task"]}.')
        raise Exception(f'There is no task called {results["task"]}.')
    elif results['task'] == 'legal_judgment_prediction':
        results['summarization'] = config.get('common', 'summarization')

        if results['summarization'] not in check_table['summarizations']:
            logger.error(
                f'There is no summarization called {results["summarization"]}.')
            raise Exception(
                f'There is no summarization called {results["summarization"]}.')
        elif results['summarization'] == 'none':
            results['article_lowerbound'] = \
                config.getint('common', 'article_lowerbound')
            results['parameters'] = config.get('common', 'parameters')

            with open(
                    file=results['parameters']
                    , mode='rb') as pkl_file:
                parameters = pickle.load(file=pkl_file)
                pkl_file.close()

            results['articles_times_appeared_of_all_files'] = \
                parameters['articles_times_appeared_of_all_files']
            results['article_sources_times_appeared_of_all_files'] = \
                parameters['article_sources_times_appeared_of_all_files']
            results['accusations_times_appeared_of_all_files'] = \
                parameters['accusations_times_appeared_of_all_files']
        elif results['summarization'] == 'bart':
            gpus = check_and_set_gpus(device_str=device_str)

            results['model_name'] = config.get('model', 'model_name')

            if results['model_name'] not in check_table['model_names']:
                logger.error(
                    f'There is no model_name called {results["model_name"]}.')
                raise Exception(
                    f'There is no model_name called {results["model_name"]}.')

            results['model'] = initialize_model(results['model_name'])(config)
            results['model'].cuda()

            try:
                logger.info('Start to initialize multiple gpus.')

                results['model'].initialize_multiple_gpus(gpus)

                logger.info('Initialize multiple gpus successfully.')
            except Exception:
                logger.warning('Failed to initialize multiple gpus.')

            if checkpoint_path is None:
                logger.error('The checkpoint path is none.')
                raise Exception('The checkpoint path is none.')

            parameters = torch.load(checkpoint_path)
            results['model'].load_state_dict(parameters['model'])
            results['model'].eval()

            results['formatter'] = initialize_formatter(config=config)

    if results['type'] not in check_table['types']:
        logger.error(f'There is no type called {results["type"]}.')
        raise Exception(f'There is no type called {results["type"]}.')

    if not os.path.exists(path=results['data_path']):
        logger.error(
            f'The path of data_path {results["data_path"]} does not exist.')
        raise Exception(
            f'The path of data_path {results["data_path"]} does not exist.')

    if not os.path.exists(path=results['output_path']):
        logger.warn(
            f'The path of output {results["output_path"]} does not exist.')
        logger.info('Make the directory automatically.')

        os.makedirs(name=results['output_path'])

    logger.info('Initialize successfully.')

    return results


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