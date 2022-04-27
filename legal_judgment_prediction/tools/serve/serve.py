import logging
import torch
import socket

from legal_judgment_prediction.tools.serve.utils import get_table, encode_data


logger = logging.getLogger(__name__)


def serve_simple_IO(parameters, config, gpu_list):
    model = parameters['model']
    model.eval()

    logger.info('Begin to get tables...')

    charge_table, article_source_table, article_table = get_table(config, mode='serve')

    logger.info('Get tables done...')

    while True:
        fact = input('Enter a fact: ')

        if fact == 'exit':
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
    model = parameters['model']
    model.eval()

    result = []

    logger.info('Begin to get tables...')

    charge_table, article_source_table, article_table = get_table(config, mode='serve')

    logger.info('Get tables done...')

    # socket server
    model_HOST, model_PORT = '172.17.0.3', 8000

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((model_HOST, model_PORT))
    server.listen(10)

    while True:
        conn, addr = server.accept()
        clientMessage = str(conn.recv(1024), encoding='utf-8')

        print(f'Connected. clientMessage: {clientMessage}')

        fact = clientMessage
        fact = encode_data(config, mode='serve', data=fact, data_name='fact')

        result = model(fact, config, gpu_list, acc_result=None, mode='serve')

        # the size of charge_result = [number_of_class]
        charge_result = torch.max(result['accuse'], 2)[1]
        article_source_result = torch.max(result['article_source'], 2)[1]
        article_result = torch.max(result['article'], 2)[1]
        
        reply_text = ''

        for key, value in charge_table.items():
            if torch.equal(charge_result, value):
                reply_text += (f'The charge of this fact: {key}' + '\n')
                break

        for key, value in article_source_table.items():
            if torch.equal(article_source_result, value):
                reply_text += (f'The article_source of this fact: {key}' + '\n')
                break

        for key, value in article_table.items():
            if torch.equal(article_result, value):
                reply_text += (f'The article of this fact: {key}' + '\n')
                break
        
        serverMessage = reply_text
        conn.sendall(serverMessage.encode())
        conn.close()