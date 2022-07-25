import logging
import torch
import socket
import time

from legal_judgment_prediction.tools.serve.utils import get_table, encode_data, Server_Thread


logger = logging.getLogger(__name__)


def serve_simple_IO(parameters, config):
    model_name = parameters['model_name']
    model = parameters['model']

    model.eval()

    if model_name == 'LJPBart':
        while True:
            fact = input('Enter a fact: ')

            logger.info(f'The input fact: {fact}')

            if fact == 'shutdown':
                break

            fact = encode_data(config, data={'fact': fact}, mode='serve', model_name=model_name)
            # fact = encode_data(config, mode='serve', data=fact, data_name='fact')

            result = model(config, fact, mode='serve', acc_result=None)

            print(f'The article of this fact: {result}')
            logger.info(f'The article of this fact: {result}')

            print()
    elif model_name == 'LJPBert':
        logger.info('Begin to get tables...')

        accusation_table, article_source_table, article_table = get_table(config, mode='serve', model_name=model_name)

        logger.info('Get tables done...')

        while True:
            fact = input('Enter a fact: ')

            logger.info(f'The input fact: {fact}')

            if fact == 'shutdown':
                break

            fact = encode_data(config, data={'fact': fact}, mode='serve', model_name=model_name)

            result = model(config, fact, mode='serve', acc_result=None)

            # The size of accusation_result = [number_of_class]
            accusation_result = torch.max(result['accusation'], 2)[1]
            article_source_result = torch.max(result['article_source'], 2)[1]
            article_result = torch.max(result['article'], 2)[1]

            for key, value in accusation_table.items():
                if torch.equal(accusation_result, value):
                    print(f'The accusation of this fact: {key}')
                    logger.info(f'The accusation of this fact: {key}')
                    break

            for key, value in article_source_table.items():
                if torch.equal(article_source_result, value):
                    print(f'The article_source of this fact: {key}')
                    logger.info(f'The article_source of this fact: {key}')
                    break

            for key, value in article_table.items():
                if torch.equal(article_result, value):
                    print(f'The article of this fact: {key}')
                    logger.info(f'The article of this fact: {key}')
                    break

            print()
    else:
        logger.error(f'There is no model_name named {model_name}.')
        raise Exception(f'There is no model_name named {model_name}.')


def serve_socket(parameters, config):
    counter = 0

    while True:
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((parameters['server_socket_IP'], parameters['server_socket_port']))
            server_socket.listen(5)

            information = 'Socket server launched.'
            logger.info(information)

            break
        except:
            error = 'Socket server launched failed.'
            logger.error(error)

            if counter < 3:
                counter += 1
                time.sleep(3)
            else:
                raise error

    server_thread = Server_Thread(server_socket=server_socket, parameters=parameters, config=config)

    counter = 0

    while True:
        try:
            server_thread.start()

            information = 'Server thread launched.'
            logger.info(information)

            break
        except:
            error = 'Server thread launched failed.'
            logger.error(error)

            if counter < 3:
                counter += 1
                time.sleep(3)
            else:
                raise error

    server_thread.join()

    information = 'Server socket closed.'
    logger.info(information)