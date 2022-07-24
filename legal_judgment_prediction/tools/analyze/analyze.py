import logging
import pickle

from legal_judgment_prediction.tools.analyze.initialize import initialize_all
from legal_judgment_prediction.tools.analyze.files_analysis import files_analysis
from legal_judgment_prediction.tools.analyze.general_analysis import general_analysis


logger = logging.getLogger(__name__)


def analyze(config, label):
    parameters = initialize_all(config, label)
    data = {}

    whole_dataset_length_of_fact_in_each_data, whole_dataset_times_appeared_of_relevant_article_sources, whole_dataset_times_appeared_of_relevant_articles, whole_dataset_times_appeared_of_accusations, data = files_analysis(parameters, data)
    general_analysis(parameters, whole_dataset_length_of_fact_in_each_data, whole_dataset_times_appeared_of_relevant_article_sources, whole_dataset_times_appeared_of_relevant_articles, whole_dataset_times_appeared_of_accusations)

    with open(file=parameters['parameters_file_path'], mode='wb') as file:
        pickle.dump(data, file)