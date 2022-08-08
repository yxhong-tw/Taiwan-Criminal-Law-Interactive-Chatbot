import logging
import os
import json
import datetime
import pickle

from utils import string_process


logger = logging.getLogger(__name__)


def files_analyze(parameters):
    logger.info('Start to analyze all files.')

    files_concepts = []

    concepts = [
        '----- Files Analyzing Task -----\n'
        , f'Dataset name: {parameters["name"]}\n'
        , f'Current time: {str(datetime.datetime.now())}\n'
        , 'The concepts of files:\n'
    ]

    files_concepts.append(concepts)

    if parameters['type'] == 'CAIL2020_sfzy':
        results, files_concepts = cail2020_sfzy_file_analyze(parameters, files_concepts)
    elif parameters['type'] == 'taiwan_indictments':
        results, files_concepts = \
            taiwan_indictments_files_analyze(parameters, files_concepts)
    elif parameters['type'] == 'CNewSum_v2':
        results, files_concepts = \
            cnewsum_v2_files_analyze(parameters, files_concepts)

    logger.info('Start to write results.')

    with open(
            file=os.path.join(
                parameters['output_path']
                , 'files_analysis.txt')
            , mode='a'
            , encoding='UTF-8') as txt_file:
        for file_concepts in files_concepts:
            for file_concept in file_concepts:
                txt_file.write(file_concept)

        txt_file.close()

    logger.info('Write results successfully.')
    logger.info('Analyze all files successfully.')

    return results


def cail2020_sfzy_file_analyze(parameters, files_concepts):
    text_lengths_of_all_files = []
    summary_lengths_of_all_files = []

    for file_name in os.listdir(parameters['data_path']):
        if file_name == 'README.md':
            continue

        text_lengths = []
        summary_lengths = []
        data_number = 0
        max_text_length = float('-inf')
        min_text_length = float('inf')
        max_summary_length = float('-inf')
        min_summary_length = float('inf')

        with open(
                file=os.path.join(parameters['data_path'], file_name)
                , mode='r'
                , encoding='UTF-8') as json_file:
            lines = json_file.readlines()

            for line in lines:
                data = json.loads(line)

                text = ''

                for sentence in data['text']:
                    text += sentence['sentence']

                text = string_process(
                    data=text, adjust_special_chars=True)

                # The length of article is 
                # calculated and saved in this part.
                # -----
                if len(text) > max_text_length:
                    max_text_length = len(text)

                if len(text) < min_text_length:
                    min_text_length = len(text)

                text_lengths.append(len(text))
                text_lengths_of_all_files.append(len(text))
                # -----

                summary = string_process(
                    data=data['summary'], adjust_special_chars=True)

                # The length of summary is
                # calculated and saved in this part.
                # -----
                if len(summary) > max_summary_length:
                    max_summary_length = len(summary)

                if len(summary) < min_summary_length:
                    min_summary_length = len(summary)

                summary_lengths.append(len(summary))
                summary_lengths_of_all_files.append(len(summary))
                # -----

                data_number += 1

            json_file.close()

        logger.info('Start to generate the strings that will \
be written into results.')

        file_concepts = [
            f'\t- {file_name}\n'
            , f'\t\t- The average length of texts: \
{float(sum(text_lengths) / len(text_lengths))}\n'
            , f'\t\t- The maximum length of texts: {max_text_length}\n'
            , f'\t\t- The minimum length of texts: {min_text_length}\n'
            , f'\t\t- The average length of summaries: \
{float(sum(summary_lengths) / len(summary_lengths))}\n'
            , f'\t\t- The maximum length of summaries: {max_summary_length}\n'
            , f'\t\t- The minimum length of summaries: {min_summary_length}\n'
            , f'\t\t- The number of data in this file: {data_number}\n'
        ]

        logger.info('Generate the strings that will be written into \
results successfully.')

        files_concepts.append(file_concepts)

    results = {
        'text_lengths_of_all_files': text_lengths_of_all_files,
        'summary_lengths_of_all_files': summary_lengths_of_all_files
    }

    return results, files_concepts


def taiwan_indictments_files_analyze(parameters, files_concepts):
    fact_lengths_of_all_files = []
    articles_times_appeared_of_all_files = {}
    article_sources_times_appeared_of_all_files = {}
    accusations_times_appeared_of_all_files = {}

    for file_name in os.listdir(parameters['data_path']):
        if file_name == 'README.md':
            continue

        fact_lengths = []
        source_indictment_files_times_cited = {}
        articles_times_appeared = {}
        article_sources_times_appeared = {}
        accusations_times_appeared = {}
        articles_numbers_of_every_data = []
        criminals_times_appeared = {}
        criminals_numbers_of_every_data = []
        indexes_of_death_penalty_not_null = []
        indexes_of_imprisonment_not_null = []
        indexes_of_life_imprisonment_not_null = []
        data_number = 0
        file_concepts = []

        with open(
                file=os.path.join(parameters['data_path'], file_name)
                , mode='r'
                , encoding='UTF-8') as json_file:
            lines = json_file.readlines()

            for index, line in enumerate(lines):
                data = json.loads(line)

                # The length of fact is calculated and saved in this part.
                # -----
                fact_lengths.append(len(data['fact']))
                fact_lengths_of_all_files.append(len(data['fact']))
                # -----

                # The times cited of source indictment file
                # is calculated and save in this part.
                # -----
                if data['file'] in source_indictment_files_times_cited:
                    source_indictment_files_times_cited[data['file']] += 1
                else:
                    source_indictment_files_times_cited[data['file']] = 1
                # -----

                # The times appeared of article and article_source are
                # calculated and save in this part.
                # -----
                for relevant_article in data['meta']['relevant_articles']:
                    article = relevant_article[0] + relevant_article[1]
                    article_source = relevant_article[0]

                    if article in articles_times_appeared:
                        articles_times_appeared[article] += 1
                    else:
                        articles_times_appeared[article] = 1

                    if article in articles_times_appeared_of_all_files:
                        articles_times_appeared_of_all_files[article] += 1
                    else:
                        articles_times_appeared_of_all_files[article] = 1

                    if article_source in article_sources_times_appeared:
                        article_sources_times_appeared[article_source] += 1
                    else:
                        article_sources_times_appeared[article_source] = 1

                    if article_source in \
                            article_sources_times_appeared_of_all_files:
                        article_sources_times_appeared_of_all_files[
                            article_source] += 1
                    else:
                        article_sources_times_appeared_of_all_files[
                            article_source] = 1
                # -----

                # The times appeared of accusation is calculated and save
                # in this part.
                # -----
                accusation = data['meta']['accusation']

                if accusation in accusations_times_appeared:
                    accusations_times_appeared[accusation] += 1
                else:
                    accusations_times_appeared[accusation] = 1

                if accusation in accusations_times_appeared_of_all_files:
                    accusations_times_appeared_of_all_files[accusation] += 1
                else:
                    accusations_times_appeared_of_all_files[accusation] = 1
                # -----

                # The number of articles is saved in this part.
                # -----
                articles_numbers_of_every_data.append(
                    data['meta']['#_relevant_articles'])
                # -----

                # The times appeared of criminals are calculated and saved
                # in this part.
                # -----
                for criminal in data['meta']['criminals']:
                    if criminal in criminals_times_appeared:
                        criminals_times_appeared[criminal] += 1
                    else:
                        criminals_times_appeared[criminal] = 1
                # -----

                # The number of criminals is saved in this part.
                # -----
                criminals_numbers_of_every_data.append(
                    data['meta']['#_criminals'])
                # -----

                # The data indexes which death penalty is not null are saved
                # in this part.
                # -----
                if data['meta']['term_of_imprisonment']['death_penalty'] \
                        is not None:
                    indexes_of_death_penalty_not_null.append(index)
                # -----

                # The data indexes which imprisonment is not null are saved
                # in this part.
                # -----
                if data['meta']['term_of_imprisonment']['imprisonment'] \
                        is not None:
                    indexes_of_imprisonment_not_null.append(index)
                # -----

                # The data indexes which life imprisonment
                # is not null are save in this part.
                # -----
                if data['meta']['term_of_imprisonment'] \
                        ['life_imprisonment'] is not None:
                    indexes_of_life_imprisonment_not_null.append(index)
                # -----

                data_number += 1

            json_file.close()

        # Sort the items by value from big to small.
        # -----
        source_indictment_files_times_cited = sorted(
            source_indictment_files_times_cited.items()
            , key=lambda item:item[1]
            , reverse=True)
        articles_times_appeared = sorted(
            articles_times_appeared.items()
            , key=lambda item:item[1]
            , reverse=True)
        article_sources_times_appeared = sorted(
            article_sources_times_appeared.items()
            , key=lambda item:item[1]
            , reverse=True)
        accusations_times_appeared = sorted(
            accusations_times_appeared.items()
            , key=lambda item:item[1]
            , reverse=True)
        criminals_times_appeared = sorted(
            criminals_times_appeared.items()
            , key=lambda item:item[1]
            , reverse=True)
        # -----

        logger.info('Start to generate the strings that will \
be written into results.')

        # This is the name of this file.
        file_concepts.append(f'\t- {file_name}\n')

        # This is the average length of facts in this file.
        file_concepts.append(f'\t\t- The average length of facts: \
{float(sum(fact_lengths) / len(fact_lengths))}\n')

        # These are the times cited of source indictment files in this file.
        # Commented out, these are unused information.
        # -----
        # file_concepts.append('\t\t- The times cited of files:\n')

        # for item in source_indictment_files_times_cited:
        #     # If the value of item is 1, all values after item are all 1.
        #     if item[1] == 1:
        #         file_concepts.append(
        #             '\t\t\t- All times cited of other files: 1\n')
        #         break

        #     file_concepts.append(
        #         f'\t\t\t- {str(item[0])}: {str(item[1])}\n')
        # -----

        # These are the times appeared of articles in this file.
        # -----
        file_concepts.append(
            '\t\t- The times appeared of relevant articles:\n')

        for item in articles_times_appeared:
            # If the value of item is 1, all values after item are all 1.
            if item[1] == 1:
                file_concepts.append('\t\t\t- All times appeared of \
other relevant_articles: 1\n')
                break

            file_concepts.append(
                f'\t\t\t- {str(item[0])}: {str(item[1])}\n')
        # -----

        # These are the times appeared of article_sources in this file.
        # -----
        file_concepts.append('\t\t- The times appeared of \
relevant article_sources:\n')

        for item in article_sources_times_appeared:
            # If the value of item is 1, all values after item are all 1.
            if item[1] == 1:
                file_concepts.append('\t\t\t- All times appeared of \
other relevant_article_sources: 1\n')
                break

            file_concepts.append(
                f'\t\t\t- {str(item[0])}: {str(item[1])}\n')
        # -----

        # These are the times appeared of accusations in this file.
        # -----
        file_concepts.append('\t\t- The times appeared of accusations:\n')

        for item in accusations_times_appeared:
            # If the value of item is 1, all values after item are all 1.
            if item[1] == 1:
                file_concepts.append('\t\t\t- All times appeared of \
other accusations: 1\n')
                break

            file_concepts.append(
                f'\t\t\t- {str(item[0])}: {str(item[1])}\n')
        # -----

        # This is the average number of articles in this file
        # -----
        articles_average_number = float(
            sum(articles_numbers_of_every_data) 
            / len(articles_numbers_of_every_data))

        file_concepts.append(f'\t\t- The average number of relevant \
articles: {articles_average_number}\n')
        # -----

        # These are the times appeared of criminals in this file.
        # Commented out, these are unused information.
        # -----
#         file_concepts.append('\t\t- The times appeared of criminals:\n')

#         for item in criminals_times_appeared:
#             # If the value of item is 1, all values after item are all 1.
#             if item[1] == 1:
#                 file_concepts.append('\t\t\t- All times appeared of \
# other criminals: 1\n')
#                 break

#             file_concepts.append(
#                 f'\t\t\t- {str(item[0])}: {str(item[1])}\n')
        # -----

        # This is the average number of criminals in this file.
        # Commented out, this is unused information.
        # -----
#         criminals_average_number = float(
#             sum(criminals_numbers_of_every_data) 
#             / len(criminals_numbers_of_every_data))

#         file_concepts.append(f'\t\t- The average number of criminals: \
# {criminals_average_number}\n')
        # -----

        # These are the indexes of data which 'death_penalty' is not null
        # in this file.
        # Commented out, all data's 'death_penalty' are null.
        # -----
#         file_concepts.append('\t\t- The indexes of data \
# which \'death_penalty\' is not null:\n')

#         for item in indexes_of_death_penalty_not_null:
#             file_concepts.append(f'\t\t\t- {str(item)}\n')
        # -----

        # These are the indexes of data which 'imprisonment' is not null
        # in this file.
        # Commented out, all data's 'imprisonment' are null.
        # -----
#         file_concepts.append('\t\t- The indexes of data \
# which \'imprisonment\' is not null:\n')

#         for item in indexes_of_imprisonment_not_null:
#             file_concepts.append(f'\t\t\t- {str(item)}\n')
        # -----

        # These are the indexes of data which 'life_imprisonment' 
        # is not null in this file.
        # Commented out, all data's 'life_imprisonment' are null.
        # -----
#         file_concepts.append('\t\t- The indexes of data \
# which \'life_imprisonment\' is not null:\n')

#         for item in indexes_of_life_imprisonment_not_null:
#             file_concepts.append(f'\t\t\t- {str(item)}\n')
        # -----

        # This is the number of data in this file.
        file_concepts.append(f'\t\t- The number of data in this file: \
{data_number}\n')

        logger.info('Generate the strings that will be written into \
results successfully.')

        files_concepts.append(file_concepts)

    # Sort the items by value from big to small.
    articles_times_appeared_of_all_files = sorted(
        articles_times_appeared_of_all_files.items()
        , key=lambda item:item[1]
        , reverse=True)
    article_sources_times_appeared_of_all_files = sorted(
        article_sources_times_appeared_of_all_files.items()
        , key=lambda item:item[1]
        , reverse=True)
    accusations_times_appeared_of_all_files = sorted(
        accusations_times_appeared_of_all_files.items()
        , key=lambda item:item[1]
        , reverse=True)

    results = {
        'fact_lengths_of_all_files': fact_lengths_of_all_files,
        'articles_times_appeared_of_all_files': \
            articles_times_appeared_of_all_files,
        'article_sources_times_appeared_of_all_files': \
            article_sources_times_appeared_of_all_files,
        'accusations_times_appeared_of_all_files': \
            accusations_times_appeared_of_all_files
    }

    return results, files_concepts


def cnewsum_v2_files_analyze(parameters, files_concepts):
    article_lengths_of_all_files = []
    summary_lengths_of_all_files = []

    for file_name in os.listdir(parameters['data_path']):
        if file_name == 'README.md':
            continue

        article_lengths = []
        summary_lengths = []
        data_number = 0
        max_article_length = float('-inf')
        min_article_length = float('inf')
        max_summary_length = float('-inf')
        min_summary_length = float('inf')

        with open(
                file=os.path.join(parameters['data_path'], file_name)
                , mode='r'
                , encoding='UTF-8') as json_file:
            lines = json_file.readlines()

            for line in lines:
                data = json.loads(line)

                article = ''

                for string in data['article']:
                    article += string

                article = string_process(
                    data=article, adjust_special_chars=True)

                # The length of article is 
                # calculated and saved in this part.
                # -----
                if len(article) > max_article_length:
                    max_article_length = len(article)

                if len(article) < min_article_length:
                    min_article_length = len(article)

                article_lengths.append(len(article))
                article_lengths_of_all_files.append(len(article))
                # -----

                summary = string_process(
                    data=data['summary'], adjust_special_chars=True)

                # The length of summary is
                # calculated and saved in this part.
                # -----
                if len(summary) > max_summary_length:
                    max_summary_length = len(summary)

                if len(summary) < min_summary_length:
                    min_summary_length = len(summary)

                summary_lengths.append(len(summary))
                summary_lengths_of_all_files.append(len(summary))
                # -----

                data_number += 1

            json_file.close()

        logger.info('Start to generate the strings that will \
be written into results.')

        file_concepts = [
            f'\t- {file_name}\n'
            , f'\t\t- The average length of articles: \
{float(sum(article_lengths) / len(article_lengths))}\n'
            , f'\t\t- The maximum length of articles: {max_article_length}\n'
            , f'\t\t- The minimum length of articles: {min_article_length}\n'
            , f'\t\t- The average length of summaries: \
{float(sum(summary_lengths) / len(summary_lengths))}\n'
            , f'\t\t- The maximum length of summaries: {max_summary_length}\n'
            , f'\t\t- The minimum length of summaries: {min_summary_length}\n'
            , f'\t\t- The number of data in this file: {data_number}\n'
        ]

        logger.info('Generate the strings that will be written into \
results successfully.')

        files_concepts.append(file_concepts)

    results = {
        'article_lengths_of_all_files': article_lengths_of_all_files,
        'summary_lengths_of_all_files': summary_lengths_of_all_files
    }

    return results, files_concepts


def general_analyze(parameters, results):
    logger.info('Start to analyze whole dataset.')

    concepts = [
        '----- General Analysis Task -----\n'
        , f'Dataset name: {parameters["name"]}\n'
        , f'Current time: {str(datetime.datetime.now())}\n'
        , 'File list:\n'
    ]

    if parameters['type'] == 'CAIL2020_sfzy':
        cail2020_sfzy_general_analyze(parameters, results, concepts)
    elif parameters['type'] == 'taiwan_indictments':
        concepts = \
            taiwan_indictments_general_analyze(parameters, results, concepts)
    elif parameters['type'] == 'CNewSum_v2':
        concepts = cnewsum_v2_general_analyze(parameters, results, concepts)

    logger.info('Start to write results.')

    output_file_path = \
        os.path.join(parameters['output_path'], 'general_analysis.txt')

    with open(
            file=output_file_path
            , mode='a'
            , encoding='UTF-8') as txt_file:
        for concept in concepts:
            txt_file.write(concept)

        txt_file.close()

    logger.info('Write results successfully.')
    logger.info('Analyze whole dataset successfully.')


def cail2020_sfzy_general_analyze(parameters, results, concepts):
    file_list_strings, number_of_files = \
        get_file_list_string(parameters['data_path'])

    for file_list_string in file_list_strings:
        concepts.append(file_list_string)

    concepts.append(
        f'Totol number of this dataset files: {str(number_of_files)}\n')

    concepts.append('The architecture of data:\n')

    any_file_name = os.listdir(parameters['data_path'])[0]

    if os.listdir(parameters['data_path'])[0] == 'README.md':
        any_file_name = os.listdir(parameters['data_path'])[1]

    nodes_list_strings = []

    with open(
            file=os.path.join(parameters['data_path'], any_file_name)
            , mode='r'
            , encoding='UTF-8') as json_file:
        line = json_file.readline()
        data = json.loads(line)

        logger.info(
            'Start to traversal and save the architecture of this data.')

        nodes_list_strings = traversal_all_nodes(
            nodes_list_strings=nodes_list_strings
            , data=data
            , tab_num=1)

        for nodes_list_string in nodes_list_strings:
            concepts.append(nodes_list_string)

        logger.info('Traversal the architecture of this data successfully.')

        json_file.close()

    text_lengths_of_all_files = \
        results['text_lengths_of_all_files']
    summary_lengths_of_all_files = \
        results['summary_lengths_of_all_files']

    texts_average_length = float(
        sum(text_lengths_of_all_files) 
        / len(text_lengths_of_all_files))

    logger.info(
        'Start to generate the strings that will be written into results.')

    concepts.append(
        f'The average length of texts: {texts_average_length}\n')

    summaries_average_length = float(
        sum(summary_lengths_of_all_files)
        / len(summary_lengths_of_all_files))

    concepts.append(
        f'The average length of summaries: {summaries_average_length}\n')

    logger.info('Generate the strings that will be written into results \
successfully.')

    return concepts


def taiwan_indictments_general_analyze(parameters, results, concepts):
    fact_lengths_of_all_files = results['fact_lengths_of_all_files']
    articles_times_appeared_of_all_files = \
        results['articles_times_appeared_of_all_files']
    article_sources_times_appeared_of_all_files = \
        results['article_sources_times_appeared_of_all_files']
    accusations_times_appeared_of_all_files = \
        results['accusations_times_appeared_of_all_files']

    file_list_strings, number_of_files = \
        get_file_list_string(parameters['data_path'])

    for file_list_string in file_list_strings:
        concepts.append(file_list_string)

    concepts.append(
        f'Totol number of this dataset files: {str(number_of_files)}\n')

    concepts.append('The architecture of data:\n')

    any_file_name = os.listdir(parameters['data_path'])[0]

    if os.listdir(parameters['data_path'])[0] == 'README.md':
        any_file_name = os.listdir(parameters['data_path'])[1]

    nodes_list_strings = []

    with open(
            file=os.path.join(parameters['data_path'], any_file_name)
            , mode='r'
            , encoding='UTF-8') as json_file:
        line = json_file.readline()
        data = json.loads(line)

        logger.info(
            'Start to traversal and save the architecture of this data.')

        nodes_list_strings = traversal_all_nodes(
            nodes_list_strings=nodes_list_strings
            , data=data
            , tab_num=1)

        for nodes_list_string in nodes_list_strings:
            concepts.append(nodes_list_string)

        logger.info('Traversal the architecture of this data successfully.')

        json_file.close()

    logger.info(
        'Start to generate the strings that will be written into results.')

    # This is the average length of all facts.
    # -----
    facts_average_length = float(
        sum(fact_lengths_of_all_files) / len(fact_lengths_of_all_files))

    concepts.append(
        f'The average length of facts: {facts_average_length}\n')
    # -----

    # These are the times appeared of all articles.
    # -----
    concepts.append('The times appeared of relevant articles:\n')

    for item in articles_times_appeared_of_all_files:
        # If the value of item is 1, all values after item are all 1.
        if item[1] == 1:
            concepts.append(
                '\t- All times appeared of other relevant_articles: 1\n')
            break

        concepts.append(f'\t- {str(item[0])}: {str(item[1])}\n')
    # -----

    # These are the times appeared of all article_sources.
    # -----
    concepts.append('The times appeared of relevant article_sources:\n')

    for item in article_sources_times_appeared_of_all_files:
        # If the value of item is 1, all values after item are all 1.
        if item[1] == 1:
            concepts.append('\t- All times appeared of other \
relevant_article_sources: 1\n')
            break

        concepts.append(f'\t- {str(item[0])}: {str(item[1])}\n')
    # -----

    # These are the times appeared of all accusations.
    # -----
    concepts.append('The times appeared of accusations:\n')

    for item in accusations_times_appeared_of_all_files:
        # If the value of item is 1, all values after item are all 1.
        if item[1] == 1:
            concepts.append(
                '\t- All times appeared of other accusations: 1\n')
            break
        
        concepts.append(f'\t- {str(item[0])}: {str(item[1])}\n')
    # -----

    logger.info('Generate the strings that will be written into results \
successfully.')

    return concepts


def cnewsum_v2_general_analyze(parameters, results, concepts):
    file_list_strings, number_of_files = \
        get_file_list_string(parameters['data_path'])

    for file_list_string in file_list_strings:
        concepts.append(file_list_string)

    concepts.append(
        f'Totol number of this dataset files: {str(number_of_files)}\n')

    concepts.append('The architecture of data:\n')

    any_file_name = os.listdir(parameters['data_path'])[0]

    if os.listdir(parameters['data_path'])[0] == 'README.md':
        any_file_name = os.listdir(parameters['data_path'])[1]

    nodes_list_strings = []

    with open(
            file=os.path.join(parameters['data_path'], any_file_name)
            , mode='r'
            , encoding='UTF-8') as json_file:
        line = json_file.readline()
        data = json.loads(line)

        logger.info(
            'Start to traversal and save the architecture of this data.')

        nodes_list_strings = traversal_all_nodes(
            nodes_list_strings=nodes_list_strings
            , data=data
            , tab_num=1)

        for nodes_list_string in nodes_list_strings:
            concepts.append(nodes_list_string)

        logger.info('Traversal the architecture of this data successfully.')

        json_file.close()

    article_lengths_of_all_files = \
        results['article_lengths_of_all_files']
    summary_lengths_of_all_files = \
        results['summary_lengths_of_all_files']

    articles_average_length = float(
        sum(article_lengths_of_all_files) 
        / len(article_lengths_of_all_files))

    logger.info(
        'Start to generate the strings that will be written into results.')

    concepts.append(
        f'The average length of articles: {articles_average_length}\n')

    summaries_average_length = float(
        sum(summary_lengths_of_all_files)
        / len(summary_lengths_of_all_files))

    concepts.append(
        f'The average length of summaries: {summaries_average_length}\n')

    logger.info('Generate the strings that will be written into results \
successfully.')

    return concepts


# Get file list strings from the input data path.
def get_file_list_string(data_path):
    file_list_strings = []
    number_of_files = 0

    for file_name in os.listdir(data_path):
        if file_name == 'README.md':
            continue

        file_list_strings.append(f'\t- {file_name}\n')
        number_of_files += 1

    return file_list_strings, number_of_files


# Traversal all node in this input data.
def traversal_all_nodes(nodes_list_strings, data, tab_num):
    if type(data) == dict:
        for item in data:
            tab_string = ('\t' * tab_num)
            nodes_list_strings.append(f'{tab_string}- {item}\n')
            nodes_list_strings = \
                traversal_all_nodes(nodes_list_strings, data[item], tab_num+1)

    return nodes_list_strings


def write_back_results(parameters, results):
    with open(
            file=os.path.join(parameters['output_path'], 'parameters.pkl')
            , mode='wb') as pkl_file:
        pickle.dump(results, pkl_file)

        pkl_file.close()