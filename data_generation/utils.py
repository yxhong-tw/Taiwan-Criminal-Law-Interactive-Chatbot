import logging
import os
import json
import torch

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils import get_tables, string_process


logger = logging.getLogger(__name__)


def get_summarization_data(parameters):
    logger.info('Start to get summarization data.')

    data = load_data(parameters['data_path'])
    results = []

    for one_data in data:
        one_data = json.loads(one_data)

        text = ''
        summary = one_data['summary']

        if parameters['data_name'] == 'CAIL2020_sfzy':
            for item in one_data['text']:
                text += item['sentence']
        elif parameters['data_name'] == 'CNewSum_v2':
            for item in one_data['article']:
                text += item

        result = {
            'text': string_process(data=text, adjust_special_chars=True)
            , 'summary': string_process(data=summary, adjust_special_chars=True)
        }

        results.append(json.dumps(result, ensure_ascii=False) + '\n')

    write_back_results(parameters, results)

    logger.info('Get summarization data successfully.')


def top_50_articles(parameters):
    logger.info(f'Start to get the data in {parameters["range"]} range.')

    change_mode_to_all = False

    # It is possible one label is top 50 but another label is not.
    # So if 'data_type' equals to 'multi_labels'
    # , use 'all' function no matter what user choose.
    if parameters['label'] == 'multi_labels':
        parameters['range'] = 'all_article'

        logger.info(f'Because the label of dataset is {parameters["label"]}, \
changing the range to {parameters["range"]}.')

        all(parameters)

        change_mode_to_all = True
    elif parameters['label'] == 'one_label':
        data = []
        articles_table = {}
        article_sources_table = {}
        accusations_table = {}
        
        for file_name in os.listdir(parameters['data_path']):
            if file_name == 'README.md':
                continue

            with open(
                    file=os.path.join(parameters['data_path'], file_name)
                    , mode='r'
                    , encoding='UTF-8') as json_file:
                lines = json_file.readlines()

                for line in lines:
                    item = json.loads(line)

                    # In reality, len(item['meta']['relevant_articles'])
                    # will be 1 in this mode.
                    for relevant_article in item['meta']['relevant_articles']:
                        article = relevant_article[0] + relevant_article[1]
                        article_source = relevant_article[0]
                        accusation = item['meta']['accusation']

                        for this_article in \
                            parameters['articles_times_appeared_of_all_files']:

                            if int(this_article[1]) < 50:
                                break
                        
                            if article == this_article[0]:
                                data.append(line)

                                if article not in articles_table and \
                                        article != '':
                                    articles_table[article] = 1

                                if article_source not in \
                                        article_sources_table and \
                                        article_source != '':
                                    article_sources_table[article_source] = 1

                                if accusation not in accusations_table and \
                                        accusation != '':
                                    accusations_table[accusation] = 1

    if change_mode_to_all == False:
        write_back_results(
            parameters
            , data
            , articles_table
            , article_sources_table
            , accusations_table
        )

        logger.info(
            f'Get the data in {parameters["range"]} range successfully.')


def all(parameters):
    logger.info(f'Start to get data in {parameters["range"]} range.')

    data = []
    articles_table = {}
    article_sources_table = {}
    accusations_table = {}
    
    for file_name in os.listdir(parameters['data_path']):
        if file_name == 'README.md':
            continue

        with open(
                file=os.path.join(parameters['data_path'], file_name)
                , mode='r'
                , encoding='UTF-8') as json_file:
            lines = json_file.readlines()

            for line in lines:
                data.append(line)

                item = json.loads(line)

                for relevant_article in item['meta']['relevant_articles']:
                    article = relevant_article[0] + relevant_article[1]
                    article_source = relevant_article[0]
                    accusation = item['meta']['accusation']

                    if article not in articles_table and article != '':
                        articles_table[article] = 1

                    if article_source not in article_sources_table and \
                            article_source != '':
                        article_sources_table[article_source] = 1

                    if accusation not in accusations_table and accusation != '':
                        accusations_table[accusation] = 1

    if parameters['label'] == 'one_label':
        write_back_results(
            parameters
            , data
            , articles_table
            , article_sources_table
            , accusations_table
        )
    elif parameters['label'] == 'multi_labels':
        write_back_results(
            parameters
            , data
        )

    logger.info(f'Get data in {parameters["range"]} range successfully.')


def get_innocence_data(parameters):
    logger.info('Start to get innocence data.')

    data = []

    articles_table, article_sources_table, accusations_table = \
        get_tables(parameters['config'], parameters['formatter'])

    for file_name in os.listdir(parameters['data_path']):
        if file_name == 'README.md':
            continue

        logger.info(f'Start to process {file_name}.')

        data_number = 0

        with open(
                file=os.path.join(parameters['data_path'], file_name)
                , mode='r'
                , encoding='UTF-8') as jsonl_file:
            lines = jsonl_file.readlines()

            for line in tqdm(lines):
                item = json.loads(line)

                fact = ''

                for string in item['article']:
                    fact += string

                fact = string_process(data=fact, adjust_special_chars=True)

                fact_tensor = parameters['formatter'](fact)

                result = parameters['model'](
                    parameters['config']
                    , fact_tensor
                    , mode='generate'
                    , acc_result=None)

                article_result = torch.max(result['article'], 2)[1]
                article_source_result = \
                    torch.max(result['article_source'], 2)[1]
                accusation_result = torch.max(result['accusation'], 2)[1]

                no_article_result = True
                no_article_source_result = True
                no_accusation_result = True

                for key, value in articles_table.items():
                    if torch.equal(article_result, value):
                        no_article_result = False
                        break

                for key, value in article_sources_table.items():
                    if torch.equal(article_source_result, value):
                        no_article_source_result = False
                        break

                for key, value in accusations_table.items():
                    if torch.equal(accusation_result, value):
                        no_accusation_result = False
                        break

                if no_article_result == False and \
                        no_article_source_result == False and \
                        no_accusation_result == False:
                    data.append(line)
                    data_number += 1

        logger.info(f'Process {file_name} successfully.')
        logger.info(f'{data_number} of {len(lines)} in {file_name} \
were judged innocent.')

    data = convert_CNewSum_data_format(data)

    write_back_results(parameters, data)

    logger.info('Get innocence data successfully.')


def combine_data(parameters):
    logger.info('Start to combine crime and innocence data.')

    data = load_data(parameters['crime_data_path'])
    innocence_data = load_data(parameters['innocence_data_path'])

    for one_data in innocence_data:
        data.append(one_data)

    write_back_results(parameters, data)

    logger.info('Combine crime and innocence data successfully.')


def load_data(data_path):
    logger.info('Start to load data.')

    data = []

    for file_name in os.listdir(data_path):
        if file_name == 'README.md':
            continue

        logger.info(f'Start to process {file_name}.')

        with open(
                file=os.path.join(data_path, file_name)
                , mode='r'
                , encoding='UTF-8') as file:
            lines = file.readlines()

            for line in lines:
                data.append(line)

            file.close()

        logger.info(f'Process {file_name} successfully.')

    logger.info('Load data successfully.')

    return data


def convert_CNewSum_data_format(lines):
    logger.info('Start to convert the format of CNewSum data.')

    data = []

    for line in lines:
        one_data = json.loads(line)

        one_data = {
            'fact': one_data['summary']
            , 'file': ''
            , 'meta': {
                'relevant_articles': []
                , '#_relevant_articles': 0
                , 'relevant_articles_org': []
                , 'accusation': '无罪'
                , 'criminals': []
                , '#_criminals': 0
                , 'term_of_imprisonment': {
                    'death_penalty': None
                    , 'imprisonment': None
                    , 'life_imprisonment': None
                }
            }
        }

        one_data = (json.dumps(one_data, ensure_ascii=False) + '\n')
        data.append(f'{one_data}\n')

    logger.info('Convert the format of CNewSum data successfully.')

    return data


def write_back_results(
        parameters
        , results
        , articles_table=None
        , article_sources_table=None
        , accusations_table=None):
    logger.info('Start to write back results.')
    
    if articles_table is not None \
            or article_sources_table is not None \
            or accusations_table is not None:
        write_aaa_data(
            output_path=parameters['output_path']
            , articles_table=articles_table
            , article_sources_table=article_sources_table
            , accusations_table=accusations_table
        )

    generate_test_data = True
    
    if parameters['type'] == 'summarization':
        generate_test_data = False

    write_tvt_data(
        parameters=parameters
        , results=results
        , output_path=parameters['output_path']
        , generate_test_data=generate_test_data
    )

    logger.info('Write back results successfully.')


def write_tvt_data(
        parameters
        , results
        , output_path
        , generate_test_data):
    train_data, valid_data = train_test_split(
        results
        , random_state=parameters['random_seed']
        , train_size=parameters['train_size'])

    file_names_to_data = {'train.json': train_data, 'valid.json': valid_data}

    if generate_test_data is True:
        valid_data, test_data = train_test_split(
            valid_data
            , random_state=parameters['random_seed']
            , train_size=parameters['valid_size'])

        file_names_to_data['valid.json'] = valid_data
        file_names_to_data['test.json'] = test_data

    for file_name in file_names_to_data:
        file_path = os.path.join(output_path, file_name)

        logger.info(f'Start to write {file_name}.')

        with open(
                file=file_path
                , mode='w'
                , encoding='UTF-8') as json_file:
            for data in file_names_to_data[file_name]:
                json_file.write(f'{data}')

            json_file.close()

        logger.info(f'Write {file_name} successfully.')


def write_aaa_data(
        output_path
        , articles_table
        , article_sources_table
        , accusations_table):
    table_names_to_data = {
        'articles.txt': articles_table
        , 'article_sources.txt': article_sources_table
        , 'accusations.txt': accusations_table
    }

    for table_name in table_names_to_data:
        logger.info(f'Start to write {table_name}.')

        table_path = os.path.join(output_path, table_name)

        with open(
                file=table_path
                , mode='w'
                , encoding='UTF-8') as txt_file:
            for data in table_names_to_data[table_name]:
                txt_file.write(f'{data}\n')

            txt_file.write('others\n')
            txt_file.close()

        logger.info(f'Write {table_name} successfully.')