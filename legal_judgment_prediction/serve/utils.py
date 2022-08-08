import logging
import threading
import time
import torch

# from legal_judgment_prediction.utils import get_tables
from utils import get_tables


logger = logging.getLogger(__name__)
is_shutdown = False


class ServerThread(threading.Thread):
    def __init__(self, server_socket, parameters, config, *args, **kwargs):
        threading.Thread.__init__(self)

        self.server_socket = server_socket
        self.parameters = parameters
        self.config = config


    def run(self, *args, **kwargs):
        global is_shutdown

        client_index = 0
        client_thread_list = []

        while is_shutdown == False:
            client_socket, client_address = self.server_socket.accept()

            client_thread = ClientThread(
                client_socket
                , client_index
                , self.parameters
                , self.config)

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
                    time.sleep(3)
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
            , config
            , *args
            , **kwargs):
        threading.Thread.__init__(self)

        self.client_socket = client_socket
        self.client_index = client_index
        self.parameters = parameters
        self.config = config


    def run(self, *args, **kwargs):
        global is_shutdown
        
        if self.client_index == 0:
            model_name = self.parameters['model_name']
            model = self.parameters['model']

            if model_name == 'LJPBart':
                counter = 0

                while is_shutdown == False:
                    try:
                        logger.info('Start to receive client message.')

                        client_message = \
                            str(self.client_socket.recv(1024), encoding='UTF-8')

                        counter = 0

                        logger.info('Receive client message successfully.')
                    except:
                        logger.error('Failed to receive client message.')

                        if counter < 3:
                            counter += 1
                            time.sleep(3)

                            continue
                        else:
                            raise Exception('Failed to receive client message.')
                            
                    if client_message == 'shutdown':
                        is_shutdown = True
                    else:
                        logger.info(f'The received message: {client_message}')

                        # fact = encode_data(
                        #     self.config
                        #     , data={'fact': client_message}
                        #     , mode='serve'
                        #     , model_name=model_name)

                        fact = self.parameters['formatter'](
                            {'fact': client_message})

                        result = model(
                            self.config
                            , fact
                            , mode='serve'
                            , acc_result=None)
                
                        reply_text = ''
                        reply_text += (f'可能觸犯的法條: {result}')
                        
                        if reply_text == '':
                            reply_text = '無相應結果，請以更完整的敘述再試一次！'

                        logger.info(f'The return message: {reply_text}')
                        
                        self.client_socket.sendall(reply_text.encode())
            elif model_name == 'LJPBert':
                # accusation_table, article_source_table, article_table = \
                #     get_tables(self.config, mode='serve', model_name=model_name)

                accusation_table, article_source_table, article_table = \
                    get_tables(self.config, self.parameters['formatter'])

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
                            time.sleep(3)

                            continue
                        else:
                            raise Exception('Client message received failed.')
                            
                    if client_message == 'shutdown':
                        is_shutdown = True
                    else:
                        logger.info(f'The received message: {client_message}')

                        # fact = encode_data(
                        #     self.config
                        #     , data={'fact': client_message}
                        #     , mode='serve'
                        #     , model_name=model_name)

                        fact = self.parameters['formatter'](
                            {'fact': client_message})

                        result = model(
                            self.config
                            , fact
                            , mode='serve'
                            , acc_result=None)

                        # the size of accusation_result = [number_of_class]
                        article_result = \
                            torch.max(result['article'], 2)[1]
                        article_source_result = \
                            torch.max(result['article_source'], 2)[1]
                        accusation_result = \
                            torch.max(result['accusation'], 2)[1]
                
                        reply_text = ''

                        for key, value in accusation_table.items():
                            if torch.equal(accusation_result, value):
                                reply_text += (f'可能被起訴罪名: {key}')
                                break

                        for key, value in article_source_table.items():
                            if torch.equal(article_source_result, value):
                                reply_text += '\n' + (f'可能觸犯的法源: {key}')
                                break

                        for key, value in article_table.items():
                            if torch.equal(article_result, value):
                                reply_text += '\n' + (f'可能觸犯的法條: {key}')
                                break
                        
                        if reply_text == '':
                            reply_text = '無相應結果，請以更完整的敘述再試一次！'

                        logger.info(f'The return message: {reply_text}')

                        self.client_socket.sendall(reply_text.encode())
            else:
                logger.error(f'There is no model_name called {model_name}.')
                raise Exception(f'There is no model_name called {model_name}.')

            self.client_socket.close()
        else:
            logger.error(f'The index of client {self.client_index} is invalid.')
            raise Exception(
                f'The index of client {self.client_index} is invalid.')