import logging
import pickle
import os
import torch

from legal_judgment_prediction.model import initialize_model
from legal_judgment_prediction.formatter import initialize_formatter


logger = logging.getLogger(__name__)


def initialize_all(config, gpu_list, checkpoint_path):
    logger.info('Start to initialize.')

    check_table = {
        'types': ['summarization', 'crime', 'innocence', 'combination']
        , 'data_names': ['CAIL2020_sfzy', 'CNewSum_v2']
        , 'labels': ['one_label', 'multi_labels']
        , 'ranges': ['top_50_articles', 'all']
        , 'model_names': ['LJPBert']
    }

    results = {
        'type': config.get('common', 'type')
        , 'output_path': config.get('common', 'output_path')
        , 'train_size': config.getfloat('common', 'train_size')
        , 'random_seed': config.getint('common', 'random_seed')
    }

    if results['type'] not in check_table['types']:
        logger.error(f'There is no type called {results["type"]}.')
        raise Exception(f'There is no type called {results["type"]}.')
    elif results['type'] == 'summarization':
        results['data_name'] = config.get('common', 'data_name')

        if results['data_name'] not in check_table['data_names']:
            logger.error(
                f'The name of data {results["data_name"]} does not exist.')
            raise Exception(
                f'The name of data {results["data_name"]} does not exist.')

        results['output_path'] = os.path.join(
            results['output_path'], results['type'], results['data_name'])
        results['data_path'] = os.path.join(config.get('common', 'data_path'))

        if not os.path.exists(results['data_path']):
            logger.error(
                f'The path of data {results["data_path"]} does not exist.')
            raise Exception(
                f'The path of data {results["data_path"]} does not exist.')
    elif results['type'] == 'crime':
        results['label'] = config.get('common', 'label')

        if results['label'] not in check_table['labels']:
            logger.error(f'There is no label called {results["label"]}.')
            raise Exception(f'There is no label called {results["label"]}.')

        results['range'] = config.get('common', 'range')

        if results['range'] not in check_table['ranges']:
            logger.error(f'There is no range called {results["range"]}.')
            raise Exception(f'There is no range called {results["range"]}.')

        results['output_path'] = os.path.join(
            results['output_path']
            , results['type']
            , results['label']
            , results['range'])
        
        with open(
                file=config.get('common', 'parameters'), mode='rb') as pkl_file:
            parameters = pickle.load(file=pkl_file)
            pkl_file.close()

        results['articles_times_appeared_of_all_files'] = \
            parameters['articles_times_appeared_of_all_files']
        results['article_sources_times_appeared_of_all_files'] = \
            parameters['article_sources_times_appeared_of_all_files']
        results['accusations_times_appeared_of_all_files'] = \
            parameters['accusations_times_appeared_of_all_files']

        results['data_path'] = os.path.join(config.get('common', 'data_path'))

        if not os.path.exists(results['data_path']):
            logger.error(
                f'The path of data {results["data_path"]} does not exist.')
            raise Exception(
                f'The path of data {results["data_path"]} does not exist.')

        results['valid_size'] = config.getfloat('common', 'valid_size')
    elif results['type'] == 'innocence':
        results['output_path'] = \
            os.path.join(results['output_path'], results['type'])

        results['model_name'] = config.get('model', 'model_name')

        if results['model_name'] not in check_table['model_names']:
            logger.error(
                f'There is no model_name called {results["model_name"]}.')
            raise Exception(
                f'There is no model_name called {results["model_name"]}.')

        results['model'] = \
            initialize_model(results['model_name'])(config).cuda()

        try:
            logger.info('Start to initialize multiple gpus.')

            results['model'].initialize_multiple_gpus(gpu_list)

            logger.info('Initialize multiple gpus successfully.')
        except Exception:
            logger.warning('Failed to initialize multiple gpus.')

        if checkpoint_path is None:
            logger.error('The checkpoint path is none.')
            raise Exception('The checkpoint path is none.')

        parameters = torch.load(checkpoint_path)
        results['model'].load_state_dict(parameters['model'])
        results['model'].eval()

        results['formatter'] = initialize_formatter(config, 'generate')
        results['config'] = config
        results['data_path'] = config.get('common', 'data_path')

        if os.path.exists(results['data_path']):
            logger.error(
                f'The path of data {results["data_path"]} does not exist.')
            raise Exception(
                f'The path of data {results["data_path"]} does not exist.')

        results['valid_size'] = config.getfloat('common', 'valid_size')
    elif results['type'] == 'combination':
        results['output_path'] = \
            os.path.join(results['output_path'], results['type'])
        results['crime_data_path'] = config.get('common', 'crime_data_path')

        if os.path.exists(results['crime_data_path']):
            logger.error(f'The path of data \
{results["crime_data_path"]} does not exist.')
            raise Exception(f'The path of data \
{results["crime_data_path"]} does not exist.')

        results['innocence_data_path'] = \
            config.get('common', 'innocence_data_path')

        if os.path.exists(results['innocence_data_path']):
            logger.error(f'The path of data \
{results["innocence_data_path"]} does not exist.')
            raise Exception(f'The path of data \
{results["innocence_data_path"]} does not exist.')

        results['valid_size']: config.getfloat('common', 'valid_size')

    if not os.path.exists(results['output_path']):
        logger.warn(
            f'The path of output {results["output_path"]} does not exist.')
        logger.info('Make the directory automatically.')

        os.makedirs(results['output_path'])

    logger.info('Initialize successfully.')

    return results