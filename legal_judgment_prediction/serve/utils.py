import logging
import threading
import time
import torch


logger = logging.getLogger(__name__)
is_shutdown = False


class ServerThread(threading.Thread):
    def __init__(self, server_socket, parameters, *args, **kwargs):
        super(ServerThread, self).__init__()

        self.server_socket = server_socket
        self.parameters = parameters


    def run(self, *args, **kwargs):
        global is_shutdown

        client_index = 0
        client_thread_list = []

        while is_shutdown == False:
            client_socket, client_address = self.server_socket.accept()

            client_thread = ClientThread(
                client_socket
                , client_index
                , self.parameters)

            counter = 0

            try:
                logger.info('Start to launch client thread.')

                client_thread.start()
                client_thread_list.append(client_thread)
                client_index += 1

                logger.info('Launch client thread successfully.')

                break
            except:
                logger.error('Failed to launch client thread.')

                if counter < 3:
                    counter += 1
                    time.sleep(secs=3)
                else:
                    raise Exception('Failed to launch client thread.')

        for client_thread in client_thread_list:
            client_thread.join()

        logger.info('Close all client sockets successfully.')

        self.server_socket.close()


class ClientThread(threading.Thread):
    def __init__(
            self
            , client_socket
            , client_index
            , parameters
            , *args
            , **kwargs):
        super(ClientThread, self).__init__()

        self.client_socket = client_socket
        self.client_index = client_index
        self.parameters = parameters


    def run(self, *args, **kwargs):
        global is_shutdown
        
        if self.client_index == 0:
            model = self.parameters['model']
            model_name = self.parameters['model_name']

            if model_name == 'LJPBart':
                counter = 0

                while is_shutdown == False:
                    try:
                        client_message = \
                            str(self.client_socket.recv(1024), encoding='UTF-8')

                        counter = 0
                    except:
                        logger.error('Failed to receive client message.')

                        if counter < 3:
                            counter += 1
                            time.sleep(secs=3)

                            continue
                        else:
                            raise Exception('Failed to receive client message.')
                            
                    if client_message == 'shutdown':
                        is_shutdown = True
                    else:
                        logger.info(f'The received message: {client_message}')

                        fact = self.parameters['formatter'](data=client_message)

                        result = model(data=fact, mode='serve', acc_result=None)
                
                        reply_text = ''
                        reply_text += (f'可能觸犯的法條: {result}')
                        
                        logger.info(f'The return message: {reply_text}')
                        
                        self.client_socket.sendall(reply_text.encode())
            elif model_name == 'LJPBert':
                articles_table = self.parameters['articles_table']
                article_sources_table = self.parameters['article_sources_table']
                accusations_table = self.parameters['accusations_table']

                counter = 0

                while is_shutdown == False:
                    try:
                        client_message = \
                            str(self.client_socket.recv(1024), encoding='UTF-8')

                        counter = 0
                    except:
                        logger.error('Client message received failed.')

                        if counter < 3:
                            counter += 1
                            time.sleep(secs=3)

                            continue
                        else:
                            raise Exception('Client message received failed.')
                            
                    if client_message == 'shutdown':
                        is_shutdown = True
                    else:
                        logger.info(f'The received message: {client_message}')

                        fact = self.parameters['formatter'](data=client_message)

                        result = model(data=fact, mode='serve', acc_result=None)

                        # The size of accusation_result is [number_of_class].
                        article_result = torch.max(
                            input=result['article']
                            , dim=2)[1]
                        article_source_result = torch.max(
                            input=result['article_source']
                            , dim=2)[1]
                        accusation_result = torch.max(
                            input=result['accusation']
                            , dim=2)[1]
                
                        reply_text = ''
                        reply_text = process_ljpbert_output_text(
                            output_text=reply_text
                            , table=articles_table
                            , table_name='article'
                            , result=article_result)
                        reply_text = process_ljpbert_output_text(
                            output_text=reply_text
                            , table=article_sources_table
                            , table_name='article_source'
                            , result=article_source_result)
                        reply_text = process_ljpbert_output_text(
                            output_text=reply_text
                            , table=accusations_table
                            , table_name='accusation'
                            , result=accusation_result)

                        logger.info(f'The return message: {reply_text}')

                        self.client_socket.sendall(reply_text.encode())

            self.client_socket.close()
        else:
            logger.error(f'The index of client {self.client_index} is invalid.')
            raise Exception(
                f'The index of client {self.client_index} is invalid.')


def process_ljpbert_output_text(output_text, table, table_name, result):
    table_name2chinese = {
        'article': '法條'
        , 'article_source': '法源'
        , 'accusation': '罪名'
    }

    already_output = False

    output_text += f'可能觸犯的{table_name2chinese[table_name]}: '

    for key, value in table.items():
        if torch.equal(input=result, other=value):
            if already_output == False:
                output_text += key
                already_output = True
                continue

            output_text += f'、{key}'

    if already_output == False:
        output_text += '無'

    output_text += '\n'

    return output_text