import logging
import pickle


logger = logging.getLogger(__name__)


def initialize_all(config):
    logger.info('Start to initialize.')

    results = {}
    results['label'] = config.get('common', 'label')
    results['range'] = config.get('common', 'range')
    results['data_path'] = config.get('common', 'data_path')
    results['output_path'] = config.get('common', 'output_path')
    results['train_size'] = config.getfloat('common', 'train_size')
    results['valid_size'] = config.getfloat('common', 'valid_size')
    results['random_seed'] = config.getint('common', 'random_seed')
    
    with open(file=config.get('common', 'parameters'), mode='rb') as pkl_file:
        parameters = pickle.load(file=pkl_file)
        pkl_file.close()

    results['relevant_articles_times_appeared_of_whole_dataset'] = parameters['relevant_articles_times_appeared_of_whole_dataset']
    results['relevant_article_sources_times_appeared_of_whole_dataset'] = parameters['relevant_article_sources_times_appeared_of_whole_dataset']
    results['accusations_times_appeared_of_whole_dataset'] = parameters['accusations_times_appeared_of_whole_dataset']

    logger.info('Initialize successfully.')

    return results