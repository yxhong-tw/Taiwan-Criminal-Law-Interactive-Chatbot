import logging
import socket
import time
import torch

from legal_judgment_prediction.serve.utils import \
    ServerThread, process_ljpbert_output_text


logger = logging.getLogger(__name__)


def serve_socket(parameters, *args, **kwargs):
    counter = 0

    while True:
        logger.info('Start to launch server socket.')

        try:
            server_socket = socket.socket(
                family=socket.AF_INET
                , type=socket.SOCK_STREAM)
            server_socket.bind(
                (parameters['server_socket_IP']
                , parameters['server_socket_port'])
            )
            server_socket.listen(5)

            logger.info('Launch server socket successfully.')

            break
        except:
            logger.error('Failed to launch server socket.')

            if counter < 3:
                counter += 1
                time.sleep(secs=3)
            else:
                raise Exception('Failed to launch server socket.')

    server_thread = ServerThread(
        server_socket=server_socket
        , parameters=parameters)

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
                time.sleep(secs=3)
            else:
                raise Exception('Failed to launch server thread.')

    server_thread.join()

    logger.info('Close server socket successfully.')


def serve_simple_IO(parameters, *args, **kwargs):
    logger.info('Start to serve.')

    model = parameters['model']
    model_name = parameters['model_name']

    if model_name == 'LJPBart':
        while True:
            fact = input('Enter a fact: ')
            logger.info(f'The input fact: {fact}')

            if fact == 'shutdown':
                logger.info('Stop to serve in simple_IO mode.')
                break

            fact = parameters['formatter'](data=fact)
            result = model(data=fact, mode='serve', acc_result=None)

            print(f'The article of this fact: {result}')
            logger.info(f'The article of this fact: {result}')

            print()
    elif model_name == 'LJPBert':
        articles_table = parameters['articles_table']
        article_sources_table = parameters['article_sources_table']
        accusations_table = parameters['accusations_table']

        while True:
            fact = input('Enter a fact: ')
            logger.info(f'The input fact: {fact}')

            if fact == 'shutdown':
                logger.info('Stop to serve.')
                break

            fact = parameters['formatter'](data=fact)
            result = model(data=fact, mode='serve', acc_result=None)

            # The size of accusation_result is [number_of_class].
            article_result = torch.max(input=result['article'], dim=2)[1]
            article_source_result = torch.max(
                input=result['article_source']
                , dim=2)[1]
            accusation_result = torch.max(
                input=result['accusation']
                , dim=2)[1]

            output_text = ''
            output_text = process_ljpbert_output_text(
                output_text=output_text
                , table=articles_table
                , table_name='article'
                , result=article_result)
            output_text = process_ljpbert_output_text(
                output_text=output_text
                , table=article_sources_table
                , table_name='article_source'
                , result=article_source_result)
            output_text = process_ljpbert_output_text(
                output_text=output_text
                , table=accusations_table
                , table_name='accusation'
                , result=accusation_result)

            print(output_text)