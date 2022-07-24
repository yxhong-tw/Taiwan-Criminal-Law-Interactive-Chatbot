import logging
import pickle


logger = logging.getLogger(__name__)


def initialize_all(config, label, range):
    results = {}

    results['label'] = label
    results['range'] = range

    results['data_path'] = config.get(label, 'data_path')
    
    file = open(file=config.get(label, 'parameters'), mode='rb')
    parameters = pickle.load(file=file)
    file.close()

    results['whole_dataset_times_appeared_of_relevant_article_sources'] = parameters['whole_dataset_times_appeared_of_relevant_article_sources']
    results['whole_dataset_times_appeared_of_relevant_articles'] = parameters['whole_dataset_times_appeared_of_relevant_articles']
    results['whole_dataset_times_appeared_of_accusations'] = parameters['whole_dataset_times_appeared_of_accusations']

    results['output_path'] = config.get('common', 'output_path')

    results['train_size'] = config.getfloat('common', 'train_size')
    results['valid_size'] = config.getfloat('common', 'valid_size')
    
    results['random_seed'] = 0

    return results