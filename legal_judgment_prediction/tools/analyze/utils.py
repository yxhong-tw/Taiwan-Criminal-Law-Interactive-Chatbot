import logging
import os
import json
import datetime
import pickle


logger = logging.getLogger(__name__)


def files_analysis(parameters):
    logger.info('Start to analyze all precedent files.')

    fact_length_in_each_data_of_whole_dataset = []
    relevant_articles_times_appeared_of_whole_dataset = {}
    relevant_article_sources_times_appeared_of_whole_dataset = {}
    accusations_times_appeared_of_whole_dataset = {}
    
    files_concepts_list = []

    for file_name in os.listdir(parameters['data_path']):
        if file_name == 'README.md':
            continue

        fact_length_in_each_data = []
        files_times_cited = {}
        relevant_articles_times_appeared = {}           # cat all parts of list
        relevant_article_sources_times_appeared = {}    # only catch list[0]
        accusations_times_appeared = {}
        relevant_articles_number_in_each_data = []
        criminals_times_appeared = {}
        criminals_number_in_each_data = []
        data_indexes_of_death_penalty_is_not_null = []
        data_indexes_of_imprisonment_is_not_null = []
        data_indexes_of_life_imprisonment_is_not_null = []
        data_number = 0

        file_concepts_strings = []

        with open(file=os.path.join(parameters['data_path'], file_name), mode='r', encoding='UTF-8') as json_file:
            lines = json_file.readlines()

            for index, line in enumerate(lines):
                data = json.loads(line)

                # Save the fact lengths of all data
                # -----
                fact_length_in_each_data.append(len(data['fact']))
                fact_length_in_each_data_of_whole_dataset.append(len(data['fact']))
                # -----

                # Calculate the times cited of files
                # -----
                if data['file'] in files_times_cited:
                    files_times_cited[data['file']] += 1
                else:
                    files_times_cited[data['file']] = 1
                # -----

                # Calculate the times appeared of relevant_articles and relevant_article_sources
                # -----
                for relevant_article in data['meta']['relevant_articles']:
                    if (relevant_article[0] + relevant_article[1]) in relevant_articles_times_appeared:
                        relevant_articles_times_appeared[(relevant_article[0] + relevant_article[1])] += 1
                    else:
                        relevant_articles_times_appeared[(relevant_article[0] + relevant_article[1])] = 1

                    if (relevant_article[0] + relevant_article[1]) in relevant_articles_times_appeared_of_whole_dataset:
                        relevant_articles_times_appeared_of_whole_dataset[(relevant_article[0] + relevant_article[1])] += 1
                    else:
                        relevant_articles_times_appeared_of_whole_dataset[(relevant_article[0] + relevant_article[1])] = 1

                    if relevant_article[0] in relevant_article_sources_times_appeared:
                        relevant_article_sources_times_appeared[relevant_article[0]] += 1
                    else:
                        relevant_article_sources_times_appeared[relevant_article[0]] = 1

                    if relevant_article[0] in relevant_article_sources_times_appeared_of_whole_dataset:
                        relevant_article_sources_times_appeared_of_whole_dataset[relevant_article[0]] += 1
                    else:
                        relevant_article_sources_times_appeared_of_whole_dataset[relevant_article[0]] = 1
                # -----

                # Calculate the times appeared of accusations
                # -----
                if data['meta']['accusation'] in accusations_times_appeared:
                    accusations_times_appeared[data['meta']['accusation']] += 1
                else:
                    accusations_times_appeared[data['meta']['accusation']] = 1

                if data['meta']['accusation'] in accusations_times_appeared_of_whole_dataset:
                    accusations_times_appeared_of_whole_dataset[data['meta']['accusation']] += 1
                else:
                    accusations_times_appeared_of_whole_dataset[data['meta']['accusation']] = 1
                # -----

                # Save the numbers of relevant articles of all data
                # -----
                relevant_articles_number_in_each_data.append(data['meta']['#_relevant_articles'])
                # -----

                # Calculate the times appeared of criminals
                # -----
                for criminal in data['meta']['criminals']:
                    if criminal in criminals_times_appeared:
                        criminals_times_appeared[criminal] += 1
                    else:
                        criminals_times_appeared[criminal] = 1
                # -----

                # Save the criminals numbers of all data
                # -----
                criminals_number_in_each_data.append(data['meta']['#_criminals'])
                # -----

                # Save the data indexes which death penalty is not null
                # -----
                if data['meta']['term_of_imprisonment']['death_penalty'] is not None:
                    data_indexes_of_death_penalty_is_not_null.append(index)
                # -----

                # Save the data indexes which imprisonment is not null
                # -----
                if data['meta']['term_of_imprisonment']['imprisonment'] is not None:
                    data_indexes_of_imprisonment_is_not_null.append(index)
                # -----

                # Save the data indexes which life imprisonment is not null
                # -----
                if data['meta']['term_of_imprisonment']['life_imprisonment'] is not None:
                    data_indexes_of_life_imprisonment_is_not_null.append(index)
                # -----

                data_number += 1

            json_file.close()

        # Sort the items by value of files_times_cited, relevant_articles_times_appeared, relevant_article_sources_times_appeared \
        # , accusations_times_appeared, criminals_times_appeared
        # -----
        files_times_cited = sorted(files_times_cited.items(), key=lambda item:item[1], reverse=True)
        relevant_articles_times_appeared = sorted(relevant_articles_times_appeared.items(), key=lambda item:item[1], reverse=True)
        relevant_article_sources_times_appeared = sorted(relevant_article_sources_times_appeared.items(), key=lambda item:item[1], reverse=True)
        accusations_times_appeared = sorted(accusations_times_appeared.items(), key=lambda item:item[1], reverse=True)
        criminals_times_appeared = sorted(criminals_times_appeared.items(), key=lambda item:item[1], reverse=True)
        # -----

        logger.info('Start to save the infomations that will be written into results.')

        # The name of this file
        file_concepts_strings.append(f'\t- {file_name}\n')

        # The average length of facts in this file
        file_concepts_strings.append('\t\t' + '- ' + 'The average length of facts: ' + str(int(sum(fact_length_in_each_data) / len(fact_length_in_each_data))) + '\n')

        # The times cited of precedent source files in this file (Commented out, this is unused information)
        # -----
        # file_concepts_strings.append('\t\t' + '- ' + 'The times cited of files: ' + '\n')

        # for item in files_times_cited:
        #     # If the value of this item is 1, all values after this item are all 1
        #     if item[1] == 1:
        #         file_concepts_strings.append('\t\t\t' + '- ' + 'All times cited of other files: ' + '1' + '\n')
        #         break

        #     file_concepts_strings.append('\t\t\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
        # -----

        # The times appeared of relevant articles in this file
        # -----
        file_concepts_strings.append('\t\t' + '- ' + 'The times appeared of relevant articles: ' + '\n')

        for item in relevant_articles_times_appeared:
            # If the value of this item is 1, all values after this item are all 1
            if item[1] == 1:
                file_concepts_strings.append('\t\t\t' + '- ' + 'All times appeared of other relevant_articles: ' + '1' + '\n')
                break

            file_concepts_strings.append('\t\t\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
        # -----

        # The times appeared of relevant article_sources in this file
        # -----
        file_concepts_strings.append('\t\t' + '- ' + 'The times appeared of relevant article_sources: ' + '\n')

        for item in relevant_article_sources_times_appeared:
            # If the value of this item is 1, all values after this item are all 1
            if item[1] == 1:
                file_concepts_strings.append('\t\t\t' + '- ' + 'All times appeared of other relevant_article_sources: ' + '1' + '\n')
                break

            file_concepts_strings.append('\t\t\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
        # -----

        # The times appeared of accusations in this file
        # -----
        file_concepts_strings.append('\t\t' + '- ' + 'The times appeared of accusations: ' + '\n')

        for item in accusations_times_appeared:
            # If the value of this item is 1, all values after this item are all 1
            if item[1] == 1:
                file_concepts_strings.append('\t\t\t' + '- ' + 'All times appeared of other accusations: ' + '1' + '\n')
                break

            file_concepts_strings.append('\t\t\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
        # -----

        # The average number of relevant articles in this file (Commented out, this is unused information)
        file_concepts_strings.append('\t\t' + '- ' + 'The average number of relevant articles: ' + str(float(sum(relevant_articles_number_in_each_data) / len(relevant_articles_number_in_each_data))) + '\n')

        # The times appeared of criminals in this file (Commented out, too many unused information)
        # -----
        # file_concepts_strings.append('\t\t' + '- ' + 'The times appeared of criminals: ' + '\n')

        # for item in criminals_times_appeared:
        #     # If the value of this item is 1, all values after this item are all 1
        #     if item[1] == 1:
        #         file_concepts_strings.append('\t\t\t' + '- ' + 'All times appeared of other criminals: ' + '1' + '\n')
        #         break

        #     file_concepts_strings.append('\t\t\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
        # -----

        # The average number of criminals in this file (Commented out, this is unused information)
        # file_concepts_strings.append('\t\t' + '- ' + 'The average number of criminals: ' + str(int(sum(criminals_number_in_each_data) / len(criminals_number_in_each_data))) + '\n')

        # The indexes of data which 'death_penalty' is not null in this file (Commented out, all data's 'death_penalty' are null)
        # -----
        # file_concepts_strings.append('\t\t' + '- ' + 'The indexes of data which \'death_penalty\' is not null: ' + '\n')

        # for item in data_indexes_of_death_penalty_is_not_null:
        #     file_concepts_strings.append('\t\t\t' + '- ' + str(item) + '\n')
        # -----

        # The indexes of data which 'imprisonment' is not null in this file (Commented out, all data's 'imprisonment' are null)
        # -----
        # file_concepts_strings.append('\t\t' + '- ' + 'The indexes of data which \'imprisonment\' is not null: ' + '\n')

        # for item in data_indexes_of_imprisonment_is_not_null:
        #     file_concepts_strings.append('\t\t\t' + '- ' + str(item) + '\n')
        # -----

        # The indexes of data which 'life_imprisonment' is not null in this file (Commented out, all data's 'life_imprisonment' are null)
        # -----
        # file_concepts_strings.append('\t\t' + '- ' + 'The indexes of data which \'life_imprisonment\' is not null: ' + '\n')

        # for item in data_indexes_of_life_imprisonment_is_not_null:
        #     file_concepts_strings.append('\t\t\t' + '- ' + str(item) + '\n')
        # -----

        # The number of data in this file
        file_concepts_strings.append('\t\t' + '- ' + 'The number of data in this file: ' + str(data_number) + '\n')

        logger.info('Save the infomations that will be written into results successfully.')

        files_concepts_list.append(file_concepts_strings)

    logger.info('Start to write results.')

    with open(file=os.path.join(parameters['output_path'], 'files_analysis.txt'), mode='a', encoding='UTF-8') as txt_file:
        txt_file.write('----- Files Analysing Task -----' + '\n')
        txt_file.write('Dataset name: ' + parameters['name'] + '\n')
        txt_file.write('Current time: ' + str(datetime.datetime.now()) + '\n')
        txt_file.write('The concepts of files: ' + '\n')

        for file_concepts_strings in files_concepts_list:
            for file_concepts_string in file_concepts_strings:
                txt_file.write(file_concepts_string)

        txt_file.close()

    logger.info('Write results successfully.')

    # Sort the items by value
    relevant_articles_times_appeared_of_whole_dataset = sorted(relevant_articles_times_appeared_of_whole_dataset.items(), key=lambda item:item[1], reverse=True)
    relevant_article_sources_times_appeared_of_whole_dataset = sorted(relevant_article_sources_times_appeared_of_whole_dataset.items(), key=lambda item:item[1], reverse=True)
    accusations_times_appeared_of_whole_dataset = sorted(accusations_times_appeared_of_whole_dataset.items(), key=lambda item:item[1], reverse=True)

    results = {}
    results['fact_length_in_each_data_of_whole_dataset'] = fact_length_in_each_data_of_whole_dataset
    results['relevant_articles_times_appeared_of_whole_dataset'] = relevant_articles_times_appeared_of_whole_dataset
    results['relevant_article_sources_times_appeared_of_whole_dataset'] = relevant_article_sources_times_appeared_of_whole_dataset
    results['accusations_times_appeared_of_whole_dataset'] = accusations_times_appeared_of_whole_dataset

    logger.info('Analyze all precedent files successfully.')

    return results


def general_analysis(parameters, results):
    logger.info('Start to analyze whole dataset.')

    fact_length_in_each_data_of_whole_dataset = results['fact_length_in_each_data_of_whole_dataset']
    relevant_articles_times_appeared_of_whole_dataset = results['relevant_articles_times_appeared_of_whole_dataset']
    relevant_article_sources_times_appeared_of_whole_dataset = results['relevant_article_sources_times_appeared_of_whole_dataset']
    accusations_times_appeared_of_whole_dataset = results['accusations_times_appeared_of_whole_dataset']

    any_file_name = os.listdir(parameters['data_path'])[0] if os.listdir(parameters['data_path'])[0] != 'README.md' else os.listdir(parameters['data_path'])[1]
    nodes_list_strings = []

    with open(file=os.path.join(parameters['data_path'], any_file_name), mode='r', encoding='UTF-8') as json_file:
        line = json_file.readline()

        data = json.loads(line)

        logger.info('Start to traversal the architecture of this data.')
        nodes_list_strings = traversal_all_nodes(nodes_list_strings=nodes_list_strings, data=data, tab_num=1)
        logger.info('Traversal the architecture of this data successfully.')

    logger.info('Start to save the infomations that will be written into results.')

    # The average length of facts in whole dataset
    facts_average_length_string = ('The average length of facts: ' + str(int(sum(fact_length_in_each_data_of_whole_dataset) / len(fact_length_in_each_data_of_whole_dataset))) + '\n')

    # The times appeared of relevant articles in whole dataset
    # -----
    relevant_articles_times_appeared_strings = []
    relevant_articles_times_appeared_strings.append('The times appeared of relevant articles: ' + '\n')

    for item in relevant_articles_times_appeared_of_whole_dataset:
        # If the value of this item is 1, all values after this item are all 1
        if item[1] == 1:
            relevant_articles_times_appeared_strings.append('\t' + '- ' + 'All times appeared of other relevant_articles: ' + '1' + '\n')
            break

        relevant_articles_times_appeared_strings.append('\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
    # -----

    # The times appeared of relevant article_sources in whole dataset
    # -----
    relevant_article_sources_times_appeared_strings = []
    relevant_article_sources_times_appeared_strings.append('The times appeared of relevant article_sources: ' + '\n')

    for item in relevant_article_sources_times_appeared_of_whole_dataset:
        # If the value of this item is 1, all values after this item are all 1
        if item[1] == 1:
            relevant_article_sources_times_appeared_strings.append('\t' + '- ' + 'All times appeared of other relevant_article_sources: ' + '1' + '\n')
            break

        relevant_article_sources_times_appeared_strings.append('\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
    # -----

    # The times appeared of accusations in whole dataset
    # -----
    accusations_times_appeared_strings = []
    accusations_times_appeared_strings.append('The times appeared of accusations: ' + '\n')

    for item in accusations_times_appeared_of_whole_dataset:
        # If the value of this item is 1, all values after this item are all 1
        if item[1] == 1:
            accusations_times_appeared_strings.append('\t' + '- ' + 'All times appeared of other accusations: ' + '1' + '\n')
            break
        
        accusations_times_appeared_strings.append('\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
    # -----

    logger.info('Save the infomations that will be written into results successfully.')

    file_list_strings, number_of_files = get_file_list_string(parameters['data_path'])

    logger.info('Start to write results.')

    with open(file=os.path.join(parameters['output_path'], 'general_analysis.txt'), mode='a', encoding='UTF-8') as txt_file:
        txt_file.write('----- General Analysis Task -----' + '\n')
        txt_file.write(f'Dataset name: ' + parameters['name'] + '\n')
        txt_file.write('Current time: ' + str(datetime.datetime.now()) + '\n')
        txt_file.write('File list: ' + '\n')

        for string in file_list_strings:
            txt_file.write(string)

        txt_file.write('Totol number of this dataset files: ' + str(number_of_files) + '\n')

        txt_file.write('The architecture of data: ' + '\n')

        for string in nodes_list_strings:
            txt_file.write(string)

        txt_file.write(facts_average_length_string)

        for string in relevant_articles_times_appeared_strings:
            txt_file.write(string)

        for string in relevant_article_sources_times_appeared_strings:
            txt_file.write(string)

        for string in accusations_times_appeared_strings:
            txt_file.write(string)

        txt_file.close()

    logger.info('Write results successfully.')
    logger.info('Analyze whole dataset successfully.')


def write_back_results(parameters, results):
    with open(file=os.path.join(parameters['output_path'], 'parameters.pkl'), mode='wb') as pkl_file:
        pickle.dump(results, pkl_file)


# Get file list strings
def get_file_list_string(data_path):
    string_list = []
    number_of_files = 0

    for file_name in os.listdir(data_path):
        if file_name == 'README.md':
            continue

        string_list.append(f'\t- {file_name}\n')
        number_of_files += 1

    return string_list, number_of_files


# Traversal all node in this data
def traversal_all_nodes(nodes_list_strings, data, tab_num):
    if type(data) == dict:
        for item in data:
            nodes_list_strings.append(('\t' * tab_num + '- ' + item + '\n'))
            nodes_list_strings = traversal_all_nodes(nodes_list_strings, data[item], tab_num+1)

    return nodes_list_strings