import logging
import torch
import socket
import time

# from legal_judgment_prediction.utils import get_tables
from utils import get_tables
from legal_judgment_prediction.serve.utils import ServerThread


logger = logging.getLogger(__name__)


def serve_simple_IO(parameters, config, *args, **kwargs):
    logger.info('Start to serve in simple_IO mode.')

    model_name = parameters['model_name']
    model = parameters['model']

    if model_name == 'LJPBart':
        while True:
            fact = input('Enter a fact: ')
            logger.info(f'The input fact: {fact}')

            if fact == 'shutdown':
                logger.info('Stop to serve in simple_IO mode.')
                break

            # fact = encode_data(
            #     config
            #     , data={'fact': fact}
            #     , mode='serve'
            #     , model_name=model_name)

            fact = parameters['formatter']({'fact': fact})
            result = model(config, fact, mode='serve', acc_result=None)

            print(f'The article of this fact: {result}')
            logger.info(f'The article of this fact: {result}')

            print()
    elif model_name == 'LJPBert':
        # article_table, article_source_table, accusation_table = \
        #     get_tables(config, mode='serve', model_name=model_name)

        article_table, article_source_table, accusation_table = \
            get_tables(config, parameters['formatter'])

        while True:
            fact = input('Enter a fact: ')
            logger.info(f'The input fact: {fact}')

            if fact == 'shutdown':
                logger.info('Stop to serve in simple_IO mode.')
                break

            # fact = encode_data(
            #     config
            #     , data={'fact': fact}
            #     , mode='serve'
            #     , model_name=model_name)

            fact = parameters['formatter']({'fact': fact})
            result = model(config, fact, mode='serve', acc_result=None)

            # The size of accusation_result = [number_of_class].
            article_result = torch.max(result['article'], 2)[1]
            article_source_result = torch.max(result['article_source'], 2)[1]
            accusation_result = torch.max(result['accusation'], 2)[1]

            for key, value in article_table.items():
                if torch.equal(article_result, value):
                    print(f'The article of this fact: {key}')
                    logger.info(f'The article of this fact: {key}')
                    break

            for key, value in article_source_table.items():
                if torch.equal(article_source_result, value):
                    print(f'The article_source of this fact: {key}')
                    logger.info(f'The article_source of this fact: {key}')
                    break

            for key, value in accusation_table.items():
                if torch.equal(accusation_result, value):
                    print(f'The accusation of this fact: {key}')
                    logger.info(f'The accusation of this fact: {key}')
                    break

            print()


def serve_socket(parameters, config, *args, **kwargs):
    counter = 0

    while True:
        logger.info('Start to launch server socket.')

        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind(
                parameters['server_socket_IP']
                , parameters['server_socket_port'])
            server_socket.listen(5)

            logger.info('Launch server socket successfully.')

            break
        except:
            logger.error('Failed to launch server socket.')

            if counter < 3:
                counter += 1
                time.sleep(3)
            else:
                raise Exception('Failed to launch server socket.')

    server_thread = ServerThread(
        server_socket=server_socket
        , parameters=parameters, config=config)

    counter = 0

    while True:
        logger.info('Start to launch server thread.')

        try:
            server_thread.start()

            logger.info('Launch server thread successfully.')
            break
        except:
            logger.error('Failed to launch server thread.')

            if counter < 3:
                counter += 1
                time.sleep(3)
            else:
                raise Exception('Failed to launch server thread.')

    server_thread.join()

    logger.info('Close server socket successfully.')