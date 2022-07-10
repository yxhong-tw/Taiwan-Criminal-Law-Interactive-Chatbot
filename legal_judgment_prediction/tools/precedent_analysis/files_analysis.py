import logging
import os
import json
import datetime


logger = logging.getLogger(__name__)


def files_analysis(config):
    logger.info('Start to analyze files.')

    name = config.get('data', 'name')
    folder_path = config.get('data', 'folder_path')
    files_analysis_file_path = config.get('result', 'files_analysis_file_path')

    whole_dataset_length_of_fact_in_each_data = []
    whole_dataset_times_appeared_of_relevant_articles = {}
    whole_dataset_times_appeared_of_accusations = {}

    files_concepts_list = []

    for file_name in os.listdir(folder_path):
        if file_name == 'README.md':
            continue

        length_of_fact_in_each_data = []
        times_cited_of_files = {}
        times_appeared_of_relevant_articles = {}    # only catch list[0]
        number_of_relevant_articles_in_each_data = []
        times_appeared_of_accusations = {}
        times_appeared_of_criminals = {}
        number_of_criminals_in_each_data = []
        data_indexes_of_death_penalty_is_not_null = []
        data_indexes_of_imprisonment_is_not_null = []
        data_indexes_of_life_imprisonment_is_not_null = []
        number_of_data = 0

        file_concepts_strings = []

        with open(str(os.path.join(folder_path, file_name)), 'r', encoding='UTF-8') as json_file:
            lines = json_file.readlines()

            for index, line in enumerate(lines):
                data = json.loads(line)

                # Save the fact length of all data
                length_of_fact_in_each_data.append(len(data['fact']))
                whole_dataset_length_of_fact_in_each_data.append(len(data['fact']))

                # Calculate the times cited of files
                # -----
                if data['file'] in times_cited_of_files:
                    times_cited_of_files[data['file']] += 1
                else:
                    times_cited_of_files[data['file']] = 1
                # -----

                # Calculate the times appeared of relevant_articles
                # -----
                for relevant_article in data['meta']['relevant_articles']:
                    if relevant_article[0] in times_appeared_of_relevant_articles:
                        times_appeared_of_relevant_articles[relevant_article[0]] += 1
                    else:
                        times_appeared_of_relevant_articles[relevant_article[0]] = 1

                    if relevant_article[0] in whole_dataset_times_appeared_of_relevant_articles:
                        whole_dataset_times_appeared_of_relevant_articles[relevant_article[0]] += 1
                    else:
                        whole_dataset_times_appeared_of_relevant_articles[relevant_article[0]] = 1
                # -----

                # Save the relevant articles number of all data
                number_of_relevant_articles_in_each_data.append(data['meta']['#_relevant_articles'])

                # Calculate the times appeared of accusation
                # -----
                if data['meta']['accusation'] in times_appeared_of_accusations:
                    times_appeared_of_accusations[data['meta']['accusation']] += 1
                else:
                    times_appeared_of_accusations[data['meta']['accusation']] = 1

                if data['meta']['accusation'] in whole_dataset_times_appeared_of_accusations:
                    whole_dataset_times_appeared_of_accusations[data['meta']['accusation']] += 1
                else:
                    whole_dataset_times_appeared_of_accusations[data['meta']['accusation']] = 1
                # -----

                # Calculate the times appeared of criminals
                # -----
                for criminal in data['meta']['criminals']:
                    if criminal in times_appeared_of_criminals:
                        times_appeared_of_criminals[criminal] += 1
                    else:
                        times_appeared_of_criminals[criminal] = 1
                # -----

                # Save the criminals number of all data
                number_of_criminals_in_each_data.append(data['meta']['#_criminals'])

                # Save the data indexes which death penalty is not null
                if data['meta']['term_of_imprisonment']['death_penalty'] is not None:
                    data_indexes_of_death_penalty_is_not_null.append(index)

                # Save the data indexes which imprisonment is not null
                if data['meta']['term_of_imprisonment']['imprisonment'] is not None:
                    data_indexes_of_imprisonment_is_not_null.append(index)

                # save the data indexes which life imprisonment is not null
                if data['meta']['term_of_imprisonment']['life_imprisonment'] is not None:
                    data_indexes_of_life_imprisonment_is_not_null.append(index)

                number_of_data += 1

            json_file.close()

        # Sort the items by value
        times_cited_of_files = sorted(times_cited_of_files.items(), key=lambda item:item[1], reverse=True)
        times_appeared_of_relevant_articles = sorted(times_appeared_of_relevant_articles.items(), key=lambda item:item[1], reverse=True)
        times_appeared_of_accusations = sorted(times_appeared_of_accusations.items(), key=lambda item:item[1], reverse=True)
        times_appeared_of_criminals = sorted(times_appeared_of_criminals.items(), key=lambda item:item[1], reverse=True)

        # The name of this file
        file_concepts_strings.append(f'\t- {file_name}\n')

        # The average length of facts in this file
        file_concepts_strings.append('\t\t' + '- ' + 'The average length of facts: ' + str(int(sum(length_of_fact_in_each_data) / len(length_of_fact_in_each_data))) + '\n')

        # The times cited of precedent source files in this file (Commented out, this is unused information)
        # -----
        # file_concepts_strings.append('\t\t' + '- ' + 'The times cited of files: ' + '\n')

        # for item in times_cited_of_files:
        #     # If the value of this item is 1, all values after this item are all 1
        #     if item[1] == 1:
        #         file_concepts_strings.append('\t\t\t' + '- ' + 'All times cited of other files: ' + '1' + '\n')
        #         break

        #     file_concepts_strings.append('\t\t\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
        # -----

        # The times appeared of relevant articles in this file
        # -----
        file_concepts_strings.append('\t\t' + '- ' + 'The times appeared of relevant articles: ' + '\n')

        for item in times_appeared_of_relevant_articles:
            # If the value of this item is 1, all values after this item are all 1
            if item[1] == 1:
                file_concepts_strings.append('\t\t\t' + '- ' + 'All times appeared of other relevant_articles: ' + '1' + '\n')
                break

            file_concepts_strings.append('\t\t\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
        # -----

        # The average number of relevant articles in this file (Commented out, this is unused information)
        file_concepts_strings.append('\t\t' + '- ' + 'The average number of relevant articles: ' + str(float(sum(number_of_relevant_articles_in_each_data) / len(number_of_relevant_articles_in_each_data))) + '\n')

        # The times appeared of accusations in this file
        # -----
        file_concepts_strings.append('\t\t' + '- ' + 'The times appeared of accusations: ' + '\n')

        for item in times_appeared_of_accusations:
            # If the value of this item is 1, all values after this item are all 1
            if item[1] == 1:
                file_concepts_strings.append('\t\t\t' + '- ' + 'All times appeared of other accusations: ' + '1' + '\n')
                break

            file_concepts_strings.append('\t\t\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
        # -----

        # The times appeared of criminals in this file (Commented out, too many unused information)
        # -----
        # file_concepts_strings.append('\t\t' + '- ' + 'The times appeared of criminals: ' + '\n')

        # for item in times_appeared_of_criminals:
        #     # If the value of this item is 1, all values after this item are all 1
        #     if item[1] == 1:
        #         file_concepts_strings.append('\t\t\t' + '- ' + 'All times appeared of other criminals: ' + '1' + '\n')
        #         break

        #     file_concepts_strings.append('\t\t\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
        # -----

        # The average number of criminals in this file (Commented out, this is unused information)
        # file_concepts_strings.append('\t\t' + '- ' + 'The average number of criminals: ' + str(int(sum(number_of_criminals_in_each_data) / len(number_of_criminals_in_each_data))) + '\n')

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
        file_concepts_strings.append('\t\t' + '- ' + 'The number of data in this file: ' + str(number_of_data) + '\n')

        files_concepts_list.append(file_concepts_strings)

    with open(file=files_analysis_file_path, mode='a', encoding='UTF-8') as result_file:
        result_file.write('----- Files Analysing Task -----' + '\n')
        result_file.write(f'Dataset name: {name}' + '\n')
        result_file.write('Current time: ' + str(datetime.datetime.now()) + '\n')
        result_file.write('The concepts of files: ' + '\n')

        for file_concepts_strings in files_concepts_list:
            for file_concepts_string in file_concepts_strings:
                result_file.write(file_concepts_string)

        result_file.close()

    # Sort the items by value
    whole_dataset_times_appeared_of_relevant_articles = sorted(whole_dataset_times_appeared_of_relevant_articles.items(), key=lambda item:item[1], reverse=True)
    whole_dataset_times_appeared_of_accusations = sorted(whole_dataset_times_appeared_of_accusations.items(), key=lambda item:item[1], reverse=True)

    logger.info('Complete files analysis.')

    return whole_dataset_length_of_fact_in_each_data, whole_dataset_times_appeared_of_relevant_articles, whole_dataset_times_appeared_of_accusations