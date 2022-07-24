import logging
import os
import json

from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


def top_50_article(parameters):
    logger.info('Start to choose the data which article is the top 50.')

    change_mode_to_all = False

    # It is possible one label is top 50 but another label is not.
    # So if 'data_type' equals to 'multi_labels', use 'all' function no matter what user choose.
    if parameters['label'] == 'multi_labels':
        logger.info('Because \'label\' == \'multi_labels\', changing parameter[\'range\'] to \'all\'.')

        parameters['range'] = 'all'
        all(parameters)

        change_mode_to_all = True
    else:
        data = []
        
        for file_name in os.listdir(parameters['data_path']):
            if file_name == 'README.md':
                continue

            with open(str(os.path.join(parameters['data_path'], file_name)), 'r', encoding='UTF-8') as json_file:
                lines = json_file.readlines()

                for line in lines:
                    item = json.loads(line)

                    for article in parameters['whole_dataset_times_appeared_of_relevant_articles']:
                        if int(article[1]) < 50:
                            break

                        # In reality, len(item['meta']['relevant_articles']) will be 1 in this mode
                        for item_article in item['meta']['relevant_articles']:
                            if (item_article[0] + item_article[1]) == article[0]:
                                # data.append(item)
                                data.append(line)

    if change_mode_to_all == False:
        write_back_results(parameters, data)


def all(parameters):
    logger.info('Start to get all data.')

    data = []
    
    for file_name in os.listdir(parameters['data_path']):
        if file_name == 'README.md':
            continue

        with open(str(os.path.join(parameters['data_path'], file_name)), 'r', encoding='UTF-8') as json_file:
            lines = json_file.readlines()

            for line in lines:
                # item = json.loads(line)

                # data.append(item)
                data.append(line)

    write_back_results(parameters, data)


def write_back_results(parameters, data):
    train_data, temp_data = train_test_split(data, random_state=parameters['random_seed'], train_size=parameters['train_size'])
    valid_data, test_data = train_test_split(temp_data, random_state=parameters['random_seed'], train_size=parameters['valid_size'])

    logger.info('Start to write train.json.')
    with open(file=os.path.join(parameters['output_path'], parameters['label'], parameters['range'], 'train.json'), mode='w', encoding='UTF-8') as file:
        for data in train_data:
            file.write(str(data))

        file.close()

    logger.info('Start to write valid.json.')
    with open(file=os.path.join(parameters['output_path'], parameters['label'], parameters['range'], 'valid.json'), mode='w', encoding='UTF-8') as file:
        for data in valid_data:
            file.write(str(data))

        file.close()

    logger.info('Start to write test.json.')
    with open(file=os.path.join(parameters['output_path'], parameters['label'], parameters['range'], 'test.json'), mode='w', encoding='UTF-8') as file:
        for data in test_data:
            file.write(str(data))

        file.close()