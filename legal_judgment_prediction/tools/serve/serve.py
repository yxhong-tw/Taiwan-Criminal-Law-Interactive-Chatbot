import logging
import torch
import socket
import time

from legal_judgment_prediction.tools.serve.utils import get_table, encode_data, Server_Thread


logger = logging.getLogger(__name__)


def serve_simple_IO(parameters, config, gpu_list):
    model = parameters['model']
    model.eval()

    logger.info('Begin to get tables...')

    charge_table, article_source_table, article_table = get_table(config, mode='serve')

    logger.info('Get tables done...')

    while True:
        fact = input('Enter a fact: ')

        if fact == 'shutdown':
            break

        fact = encode_data(config, mode='serve', data=fact, data_name='fact')

        result = model(fact, config, gpu_list, acc_result=None, mode='serve')

        # the size of charge_result = [number_of_class]
        charge_result = torch.max(result['accuse'], 2)[1]
        article_source_result = torch.max(result['article_source'], 2)[1]
        article_result = torch.max(result['article'], 2)[1]

        for key, value in charge_table.items():
            if torch.equal(charge_result, value):
                print(f'The charge of this fact: {key}')
                break

        for key, value in article_source_table.items():
            if torch.equal(article_source_result, value):
                print(f'The article_source of this fact: {key}')
                break

        for key, value in article_table.items():
            if torch.equal(article_result, value):
                print(f'The article of this fact: {key}')
                break

        print()


def serve_socket(parameters, config, gpu_list):
    server_socket_IP, server_socket_port = '172.17.0.4', 8000

    counter = 0

    while True:
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((server_socket_IP, server_socket_port))
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

    server_thread = Server_Thread(server_socket=server_socket, parameters=parameters, config=config, gpu_list=gpu_list)

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