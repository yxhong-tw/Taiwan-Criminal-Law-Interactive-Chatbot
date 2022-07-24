import logging
import datetime
import os
import json

from legal_judgment_prediction.tools.precedent_analysis.utils import get_file_list_string, traversal_all_nodes


logger = logging.getLogger(__name__)


def general_analysis(config, whole_dataset_length_of_fact_in_each_data, whole_dataset_times_appeared_of_relevant_article_sources, whole_dataset_times_appeared_of_relevant_articles, whole_dataset_times_appeared_of_accusations):
    logger.info('Start to analyze dataset.')

    name = config.get('data', 'name')
    folder_path = config.get('data', 'folder_path')
    general_analysis_file_path = config.get('result', 'general_analysis_file_path')

    file_list_strings, number_of_files = get_file_list_string(folder_path)

    any_file_name = os.listdir(folder_path)[0] if os.listdir(folder_path)[0] != 'README.md' else os.listdir(folder_path)[1]

    nodes_list_strings = []
    with open(str(os.path.join(folder_path, any_file_name)), 'r', encoding='UTF-8') as json_file:
        line = json_file.readline()

        data = json.loads(line)

        nodes_list_strings = traversal_all_nodes(nodes_list_strings=nodes_list_strings, data=data, tab_num=1)

    # The average length of facts in whole dataset
    facts_average_length_string = 'The average length of facts: ' + str(int(sum(whole_dataset_length_of_fact_in_each_data) / len(whole_dataset_length_of_fact_in_each_data))) + '\n'

    # The times appeared of relevant article_sources in whole dataset
    # -----
    relevant_article_sources_times_appeared_strings = []
    relevant_article_sources_times_appeared_strings.append('The times appeared of relevant article_sources: ' + '\n')

    for item in whole_dataset_times_appeared_of_relevant_article_sources:
        # If the value of this item is 1, all values after this item are all 1
        if item[1] == 1:
            relevant_article_sources_times_appeared_strings.append('\t' + '- ' + 'All times appeared of other relevant_article_sources: ' + '1' + '\n')
            break

        relevant_article_sources_times_appeared_strings.append('\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
    # -----

    # The times appeared of relevant articles in whole dataset
    # -----
    relevant_articles_times_appeared_strings = []
    relevant_articles_times_appeared_strings.append('The times appeared of relevant articles: ' + '\n')

    for item in whole_dataset_times_appeared_of_relevant_articles:
        # If the value of this item is 1, all values after this item are all 1
        if item[1] == 1:
            relevant_articles_times_appeared_strings.append('\t' + '- ' + 'All times appeared of other relevant_articles: ' + '1' + '\n')
            break

        relevant_articles_times_appeared_strings.append('\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
    # -----

    # The times appeared of accusations in whole dataset
    # -----
    accusations_times_appeared_strings = []
    accusations_times_appeared_strings.append('The times appeared of accusations: ' + '\n')

    for item in whole_dataset_times_appeared_of_accusations:
        # If the value of this item is 1, all values after this item are all 1
        if item[1] == 1:
            accusations_times_appeared_strings.append('\t' + '- ' + 'All times appeared of other accusations: ' + '1' + '\n')
            break
        
        accusations_times_appeared_strings.append('\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
    # -----

    with open(file=general_analysis_file_path, mode='a', encoding='UTF-8') as result_file:
        result_file.write('----- General Analysis Task -----' + '\n')
        result_file.write(f'Dataset name: {name}' + '\n')
        result_file.write('Current time: ' + str(datetime.datetime.now()) + '\n')
        result_file.write('File list: ' + '\n')

        for string in file_list_strings:
            result_file.write(string)

        result_file.write('Totol number of this dataset files: ' + str(number_of_files) + '\n')

        result_file.write('The architecture of data: ' + '\n')

        for string in nodes_list_strings:
            result_file.write(string)

        result_file.write(facts_average_length_string)

        for string in relevant_article_sources_times_appeared_strings:
            result_file.write(string)

        for string in relevant_articles_times_appeared_strings:
            result_file.write(string)

        for string in accusations_times_appeared_strings:
            result_file.write(string)

        result_file.close()

    logger.info('Complete general analysis.')