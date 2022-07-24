import pickle

from legal_judgment_prediction.tools.precedent_analysis.files_analysis import files_analysis
from legal_judgment_prediction.tools.precedent_analysis.general_analysis import general_analysis


def analyze(config):
    parameters = {}

    whole_dataset_length_of_fact_in_each_data, whole_dataset_times_appeared_of_relevant_article_sources, whole_dataset_times_appeared_of_relevant_articles, whole_dataset_times_appeared_of_accusations, parameters = files_analysis(config, parameters)
    general_analysis(config, whole_dataset_length_of_fact_in_each_data, whole_dataset_times_appeared_of_relevant_article_sources, whole_dataset_times_appeared_of_relevant_articles, whole_dataset_times_appeared_of_accusations)

    with open(config.get('result', 'parameters_file_path'), 'wb') as file:
        pickle.dump(parameters, file)