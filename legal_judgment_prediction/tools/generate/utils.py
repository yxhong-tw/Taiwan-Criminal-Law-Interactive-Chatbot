import logging
import os
import json

from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


def top_50_article(parameters):
    logger.info(f'Start to get the data in {parameters["range"]} range.')

    change_mode_to_all = False

    # It is possible one label is top 50 but another label is not.
    # So if 'data_type' equals to 'multi_labels', use 'all' function no matter what user choose.
    if parameters['label'] == 'multi_labels':
        parameters['range'] = 'all_article'

        logger.info(f'Because the label of dataset is {parameters["label"]}, changing the range to {parameters["range"]}.')

        all(parameters)

        change_mode_to_all = True
    else:
        data = []
        accusations_dict = {}
        articles_dict = {}
        article_sources_dict = {}
        
        for file_name in os.listdir(parameters['data_path']):
            if file_name == 'README.md':
                continue

            with open(file=os.path.join(parameters['data_path'], file_name), mode='r', encoding='UTF-8') as json_file:
                lines = json_file.readlines()

                for line in lines:
                    item = json.loads(line)

                    for article in parameters['relevant_articles_times_appeared_of_whole_dataset']:
                        if int(article[1]) < 50:
                            break

                        # In reality, len(item['meta']['relevant_articles']) will be 1 in this mode
                        for item_article in item['meta']['relevant_articles']:
                            if (item_article[0] + item_article[1]) == article[0]:
                                data.append(line)

                                if item['meta']['accusation'] not in accusations_dict and item['meta']['accusation'] != '':
                                    accusations_dict[item['meta']['accusation']] = 1

                                if (item['meta']['relevant_articles'][0][0] + item['meta']['relevant_articles'][0][1]) not in articles_dict and (item['meta']['relevant_articles'][0][0] + item['meta']['relevant_articles'][0][1]) != '':
                                    articles_dict[(item['meta']['relevant_articles'][0][0] + item['meta']['relevant_articles'][0][1])] = 1

                                if item['meta']['relevant_articles'][0][0] not in article_sources_dict and item['meta']['relevant_articles'][0][0] != '':
                                    article_sources_dict[item['meta']['relevant_articles'][0][0]] = 1

    logger.info(f'Get the data in {parameters["range"]} range successfully.')

    if change_mode_to_all == False:
        write_back_results(parameters, data, accusations_dict, articles_dict, article_sources_dict)


def all(parameters):
    logger.info(f'Start to get data in {parameters["range"]} range.')

    data = []
    accusations_dict = {}
    articles_dict = {}
    article_sources_dict = {}
    
    for file_name in os.listdir(parameters['data_path']):
        if file_name == 'README.md':
            continue

        with open(file=os.path.join(parameters['data_path'], file_name), mode='r', encoding='UTF-8') as json_file:
            lines = json_file.readlines()

            for line in lines:
                data.append(line)

                item = json.loads(line)

                if item['meta']['accusation'] not in accusations_dict and item['meta']['accusation'] != '':
                    accusations_dict[item['meta']['accusation']] = 1

                if (item['meta']['relevant_articles'][0][0] + item['meta']['relevant_articles'][0][1]) not in articles_dict and (item['meta']['relevant_articles'][0][0] + item['meta']['relevant_articles'][0][1]) != '':
                    articles_dict[(item['meta']['relevant_articles'][0][0] + item['meta']['relevant_articles'][0][1])] = 1

                if item['meta']['relevant_articles'][0][0] not in article_sources_dict and item['meta']['relevant_articles'][0][0] != '':
                    article_sources_dict[item['meta']['relevant_articles'][0][0]] = 1

    logger.info(f'Get data in {parameters["range"]} range successfully.')

    write_back_results(parameters, data, accusations_dict, articles_dict, article_sources_dict)


def write_back_results(parameters, data, accusations_dict, articles_dict, article_sources_dict):
    logger.info('Start to write back results.')

    if parameters['label'] == 'one_label':
        logger.info('Start to write accusations.txt.')

        with open(file=os.path.join(parameters['output_path'], parameters['label'], parameters['range'], 'accusations.txt'), mode='w', encoding='UTF-8') as txt_file:
            for item in accusations_dict:
                txt_file.write(item + '\n')

            txt_file.write('others' + '\n')
            txt_file.close()

        logger.info('Write accusations.txt successfully.')
        logger.info('Start to write articles.txt.')

        with open(file=os.path.join(parameters['output_path'], parameters['label'], parameters['range'], 'articles.txt'), mode='w', encoding='UTF-8') as txt_file:
            for item in articles_dict:
                txt_file.write(item + '\n')

            txt_file.write('others' + '\n')
            txt_file.close()

        logger.info('Write articles.txt successfully.')
        logger.info('Start to write article_sources.txt.')

        with open(file=os.path.join(parameters['output_path'], parameters['label'], parameters['range'], 'article_sources.txt'), mode='w', encoding='UTF-8') as txt_file:
            for item in article_sources_dict:
                txt_file.write(item + '\n')

            txt_file.write('others' + '\n')
            txt_file.close()

        logger.info('Write article_sources.txt successfully.')

    train_data, temp_data = train_test_split(data, random_state=parameters['random_seed'], train_size=parameters['train_size'])
    valid_data, test_data = train_test_split(temp_data, random_state=parameters['random_seed'], train_size=parameters['valid_size'])

    logger.info('Start to write train.json.')

    with open(file=os.path.join(parameters['output_path'], parameters['label'], parameters['range'], 'train.json'), mode='w', encoding='UTF-8') as json_file:
        for data in train_data:
            json_file.write(str(data))

        json_file.close()

    logger.info('Write train.json successfully.')
    logger.info('Start to write valid.json.')

    with open(file=os.path.join(parameters['output_path'], parameters['label'], parameters['range'], 'valid.json'), mode='w', encoding='UTF-8') as json_file:
        for data in valid_data:
            json_file.write(str(data))

        json_file.close()

    logger.info('Write valid.json successfully.')
    logger.info('Start to write test.json.')

    with open(file=os.path.join(parameters['output_path'], parameters['label'], parameters['range'], 'test.json'), mode='w', encoding='UTF-8') as json_file:
        for data in test_data:
            json_file.write(str(data))

        json_file.close()

    logger.info('Write test.json successfully.')
    logger.info('Write back results successfully.')
