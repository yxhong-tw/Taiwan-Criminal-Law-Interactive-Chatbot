import logging
import pickle


logger = logging.getLogger(__name__)


def initialize_all(config, data_type, mode):
    results = {}

    results['data_type'] = data_type
    results['mode'] = mode

    if data_type == 'one_label':
        results['data_path'] = config.get('common', 'one_label_path')
        file = open(file=config.get('common', 'one_label_parameters'), mode='rb')
        parameters = pickle.load(file=file)
        file.close()
    elif data_type == 'multi_labels':
        results['data_path'] = config.get('common', 'multi_labels_path')
        file = open(file=config.get('common', 'multi_labels_parameters'), mode='rb')
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