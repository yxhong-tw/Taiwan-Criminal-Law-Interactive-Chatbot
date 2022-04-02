import datetime
import os
import json


def main():
    print('Dataset analysing task is started.')

    # get config from 'config.json'
    config = get_config()

    with open(config['dataset_analysis_result_file_path'], 'a') as result_file:
        result_file.write('----- Dataset Analysing Task -----' + '\n')
        result_file.write('Dataset name: ' + config['dataset_name'] + '\n')
        result_file.write('Current time: ' + str(datetime.datetime.now()) + '\n')

        result_file.close()

    with open(config['files_analysis_result_file_path'], 'a') as result_file:
        result_file.write('----- Dataset Analysing Task -----' + '\n')
        result_file.write('Dataset name: ' + config['dataset_name'] + '\n')
        result_file.write('Current time: ' + str(datetime.datetime.now()) + '\n')

        result_file.close()

    # list all name of files in this dataset
    list_files(config)

    # list the architecture of data
    # -----
    any_file_name = os.listdir(config['dataset_folder_path'])[0]

    with open(str(os.path.join(config['dataset_folder_path'], any_file_name))) as json_file:
        line = json_file.readline()

        data = json.loads(line)

        with open(config['dataset_analysis_result_file_path'], 'a') as result_file:
            result_file.write('The architecture of data: ' + '\n')

            result_file.close()

        traversal_data(config, data, 1)
    # -----

    # analysis all files in this dataset
    analysis_files(config)

    with open(config['dataset_analysis_result_file_path'], 'a') as result_file:
        result_file.write('----- Dataset analysing task is done -----' + '\n')

        result_file.close()

    with open(config['files_analysis_result_file_path'], 'a') as result_file:
        result_file.write('----- Dataset analysing task is done -----' + '\n')

        result_file.close()

    print('Dataset analysing task is done.')


# get config from 'config.json'
def get_config():
    with open('./dataset_analysis/config.json') as json_file:
        config = json.load(json_file)

        json_file.close()

    return config


# list all name of files in this dataset
def list_files(config):
    with open(config['dataset_analysis_result_file_path'], 'a') as result_file:
        result_file.write('Files list: ' + '\n')

        number_of_files = 0

        for file_name in os.listdir(config['dataset_folder_path']):
            result_file.write('\t' + '- ' + file_name + '\n')

            number_of_files += 1

        result_file.write('Totol number of this dataset files: ' + str(number_of_files) + '\n')

        result_file.close()


# traversal all node in this data
def traversal_data(config, data, tab_num):
    if type(data) == dict:
        for item in data:
            with open(config['dataset_analysis_result_file_path'], 'a') as result_file:
                result_file.write('\t' * tab_num + '- ' + item + '\n')

                result_file.close()

            traversal_data(config, data[item], tab_num + 1)


# analysis all files in this dataset
def analysis_files(config):
    with open(config['files_analysis_result_file_path'], 'a') as result_file:
        result_file.write('The concepts of files: ' + '\n')

        result_file.close()

    whole_dataset_length_of_fact_in_each_data = []
    whole_dataset_times_appeared_of_relevant_articles = {}
    whole_dataset_times_appeared_of_accusations = {}

    for file_name in os.listdir(config['dataset_folder_path']):
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

        with open(str(os.path.join(config['dataset_folder_path'], file_name))) as json_file:
            lines = json_file.readlines()

            for index, line in enumerate(lines):
                data = json.loads(line)

                # save the fact length of all data
                length_of_fact_in_each_data.append(len(data['fact']))
                whole_dataset_length_of_fact_in_each_data.append(len(data['fact']))

                # calculate the times cited of files
                # -----
                if data['file'] in times_cited_of_files:
                    times_cited_of_files[data['file']] += 1
                else:
                    times_cited_of_files[data['file']] = 1
                # -----

                # calculate the times appeared of relevant_articles
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

                # save the relevant articles number of all data
                number_of_relevant_articles_in_each_data.append(data['meta']['#_relevant_articles'])

                # calculate the times appeared of accusation
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

                # calculate the times appeared of criminals
                # -----
                for criminal in data['meta']['criminals']:
                    if criminal in times_appeared_of_criminals:
                        times_appeared_of_criminals[criminal] += 1
                    else:
                        times_appeared_of_criminals[criminal] = 1
                # -----

                # save the criminals number of all data
                number_of_criminals_in_each_data.append(data['meta']['#_criminals'])

                # save the data indexes which death penalty is not null
                if data['meta']['term_of_imprisonment']['death_penalty'] is not None:
                    data_indexes_of_death_penalty_is_not_null.append(index)

                # save the data indexes which imprisonment is not null
                if data['meta']['term_of_imprisonment']['imprisonment'] is not None:
                    data_indexes_of_imprisonment_is_not_null.append(index)

                # save the data indexes which life imprisonment is not null
                if data['meta']['term_of_imprisonment']['life_imprisonment'] is not None:
                    data_indexes_of_life_imprisonment_is_not_null.append(index)

                number_of_data += 1

            json_file.close()

        # sort the items by value
        times_cited_of_files = sorted(times_cited_of_files.items(), key=lambda item:item[1], reverse=True)
        times_appeared_of_relevant_articles = sorted(times_appeared_of_relevant_articles.items(), key=lambda item:item[1], reverse=True)
        times_appeared_of_accusations = sorted(times_appeared_of_accusations.items(), key=lambda item:item[1], reverse=True)
        times_appeared_of_criminals = sorted(times_appeared_of_criminals.items(), key=lambda item:item[1], reverse=True)

        with open(config['files_analysis_result_file_path'], 'a') as result_file:
            # this file name
            result_file.write('\t' + '- ' + file_name + '\n')

            # the average length of facts in this file
            result_file.write('\t\t' + '- ' + 'The average length of facts: ' + str(int(sum(length_of_fact_in_each_data) / len(length_of_fact_in_each_data))) + '\n')

            # the times cited of precedent source files in this file(being commented out, this is unused information)
            # -----
            # result_file.write('\t\t' + '- ' + 'The times cited of files: ' + '\n')

            # for item in times_cited_of_files:
            #     # if the value of this item is 1, all values after this item are all 1
            #     if item[1] == 1:
            #         result_file.write('\t\t\t' + '- ' + 'All times cited of other files: ' + '1' + '\n')
            #         break

            #     result_file.write('\t\t\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
            # -----

            # the times appeared of relevant articles in this file
            # -----
            result_file.write('\t\t' + '- ' + 'The times appeared of relevant articles: ' + '\n')

            for item in times_appeared_of_relevant_articles:
                # if the value of this item is 1, all values after this item are all 1
                if item[1] == 1:
                    result_file.write('\t\t\t' + '- ' + 'All times appeared of other relevant_articles: ' + '1' + '\n')
                    break

                result_file.write('\t\t\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
            # -----

            # the average number of relevant articles in this file(being commented out, this is unused information)
            # result_file.write('\t\t' + '- ' + 'The average number of relevant articles: ' + str(int(sum(number_of_relevant_articles_in_each_data) / len(number_of_relevant_articles_in_each_data))) + '\n')

            # the times appeared of accusations in this file
            # -----
            result_file.write('\t\t' + '- ' + 'The times appeared of accusations: ' + '\n')

            for item in times_appeared_of_accusations:
                # if the value of this item is 1, all values after this item are all 1
                if item[1] == 1:
                    result_file.write('\t\t\t' + '- ' + 'All times appeared of other accusations: ' + '1' + '\n')
                    break

                result_file.write('\t\t\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
            # -----

            # the times appeared of criminals in this file(being commented out, too many unused information)
            # -----
            # result_file.write('\t\t' + '- ' + 'The times appeared of criminals: ' + '\n')

            # for item in times_appeared_of_criminals:
            #     # if the value of this item is 1, all values after this item are all 1
            #     if item[1] == 1:
            #         result_file.write('\t\t\t' + '- ' + 'All times appeared of other criminals: ' + '1' + '\n')
            #         break

            #     result_file.write('\t\t\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
            # -----

            # the average number of criminals in this file(being commented out, this is unused information)
            # result_file.write('\t\t' + '- ' + 'The average number of criminals: ' + str(int(sum(number_of_criminals_in_each_data) / len(number_of_criminals_in_each_data))) + '\n')        

            # the indexes of data which 'death_penalty' is not null in this file(being commented out, all data's 'death_penalty' are null)
            # -----
            # result_file.write('\t\t' + '- ' + 'The indexes of data which \'death_penalty\' is not null: ' + '\n')

            # for item in data_indexes_of_death_penalty_is_not_null:
            #     result_file.write('\t\t\t' + '- ' + str(item) + '\n')
            # -----

            # the indexes of data which 'imprisonment' is not null in this file(being commented out, all data's 'imprisonment' are null)
            # -----
            # result_file.write('\t\t' + '- ' + 'The indexes of data which \'imprisonment\' is not null: ' + '\n')

            # for item in data_indexes_of_imprisonment_is_not_null:
            #     result_file.write('\t\t\t' + '- ' + str(item) + '\n')
            # -----

            # the indexes of data which 'life_imprisonment' is not null in this file(being commented out, all data's 'life_imprisonment' are null)
            # -----
            # result_file.write('\t\t' + '- ' + 'The indexes of data which \'life_imprisonment\' is not null: ' + '\n')

            # for item in data_indexes_of_life_imprisonment_is_not_null:
            #     result_file.write('\t\t\t' + '- ' + str(item) + '\n')
            # -----

            # the number of data in this file
            result_file.write('\t\t' + '- ' + 'The number of data in this file: ' + str(number_of_data) + '\n')

            result_file.close()

    # sort the items by value
    whole_dataset_times_appeared_of_relevant_articles = sorted(whole_dataset_times_appeared_of_relevant_articles.items(), key=lambda item:item[1], reverse=True)
    whole_dataset_times_appeared_of_accusations = sorted(whole_dataset_times_appeared_of_accusations.items(), key=lambda item:item[1], reverse=True)

    with open(config['dataset_analysis_result_file_path'], 'a') as result_file:
        # the average length of facts in whole dataset
        result_file.write('The average length of facts: ' + str(int(sum(whole_dataset_length_of_fact_in_each_data) / len(whole_dataset_length_of_fact_in_each_data))) + '\n')

        # the times appeared of relevant articles in whole dataset
        # -----
        result_file.write('The times appeared of relevant articles: ' + '\n')

        for item in whole_dataset_times_appeared_of_relevant_articles:
            # if the value of this item is 1, all values after this item are all 1
            if item[1] == 1:
                result_file.write('\t' + '- ' + 'All times appeared of other relevant_articles: ' + '1' + '\n')
                break

            result_file.write('\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
        # -----

        # the times appeared of accusations in whole dataset
        # -----
        result_file.write('The times appeared of accusations: ' + '\n')

        for item in whole_dataset_times_appeared_of_accusations:
            # if the value of this item is 1, all values after this item are all 1
            if item[1] == 1:
                result_file.write('\t' + '- ' + 'All times appeared of other accusations: ' + '1' + '\n')
                break

            result_file.write('\t' + '- ' + str(item[0]) + ': ' + str(item[1]) + '\n')
        # -----

        result_file.close()


if __name__ == '__main__':
    main()