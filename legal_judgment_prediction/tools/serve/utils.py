import logging
import threading
import time
import torch

from legal_judgment_prediction.tools.formatter.Bert import BertLJP


logger = logging.getLogger(__name__)
is_shutdown = False


def get_table(config, mode, *args, **params):
    charge_list, article_list, article_source_list = [], [], []
    
    with open(config.get('data', 'charge_path'), 'r', encoding='utf-8') as file:
        lines = file.readlines()

        for index in range(len(lines)):
            if lines[index][-1] == '\n':
                charge_list.append(lines[index][0:-1])
            else:
                charge_list.append(lines[index])

        file.close()

    with open(config.get('data', 'article_source_path'), 'r', encoding='utf-8') as file:
        lines = file.readlines()

        for index in range(len(lines)):
            if lines[index][-1] == '\n':
                article_source_list.append(lines[index][0:-1])
            else:
                article_source_list.append(lines[index])

        file.close()

    with open(config.get('data', 'article_path'), 'r', encoding='utf-8') as file:
        lines = file.readlines()

        for index in range(len(lines)):
            if lines[index][-1] == '\n':
                article_list.append(lines[index][0:-1])
            else:
                article_list.append(lines[index])

        file.close()

    charge_table, article_source_table, article_table = {}, {}, {}

    for data in charge_list:
        charge_table[data] = encode_data(config, mode, data, data_name='charge')

    for data in article_source_list:
        article_source_table[data] = encode_data(config, mode, data, data_name='article_source')
    
    for data in article_list:
        article_table[data] = encode_data(config, mode, data, data_name='article')

    return charge_table, article_source_table, article_table


def encode_data(config, mode, data, data_name, *args, **params):
    formatter = BertLJP(config, mode, *args, **params)

    return formatter.process(data, config, mode, data_name=data_name)


class Server_Thread(threading.Thread):
    def __init__(self, server_socket, parameters, config, gpu_list):
        threading.Thread.__init__(self)

        self.server_socket = server_socket
        self.parameters = parameters
        self.config = config
        self.gpu_list = gpu_list


    def run(self):
        global is_shutdown

        client_index = 0
        client_thread_list = []

        while is_shutdown == False:
            client_socket, client_address = self.server_socket.accept()

            client_thread = Client_Thread(self.server_socket, client_socket, client_address, client_index, self.parameters, self.config, self.gpu_list)

            counter = 0

            try:
                client_thread.start()
                client_thread_list.append(client_thread)
                client_index += 1

                information = 'Client thread launched.'
                logger.info(information)

                break
            except:
                error = 'Client thread launched failed.'
                logger.error(error)

                if counter < 3:
                    counter += 1
                    time.sleep(3)
                else:
                    raise error

        for client_thread in client_thread_list:
            client_thread.join()

        information = 'All client sockets closed.'
        logger.info(information)

        self.server_socket.close()


class Client_Thread(threading.Thread):
    def __init__(self, server_socket, client_socket, client_address, client_index, parameters, config, gpu_list):
        threading.Thread.__init__(self)

        self.server_socket = server_socket
        self.client_socket = client_socket
        self.client_address = client_address
        self.client_index = client_index
        self.parameters = parameters
        self.config = config
        self.gpu_list = gpu_list


    def run(self):
        global is_shutdown
        
        if self.client_index == 0:
            model = self.parameters['model']
            model.eval()

            logger.info('Begin to get tables...')

            charge_table, article_source_table, article_table = get_table(self.config, mode='serve')

            logger.info('Get tables done...')

            counter = 0

            while is_shutdown == False:
                try:
                    client_message = str(self.client_socket.recv(1024), encoding='utf-8')

                    counter = 0
                except:
                    error = 'Client message received failed.'
                    logger.error(error)

                    if counter < 3:
                        counter += 1
                        time.sleep(3)

                        continue
                    else:
                        raise error
                        
                if client_message == 'shutdown':
                    is_shutdown = True
                else:
                    logger.info(client_message)

                    fact = encode_data(self.config, mode='serve', data=client_message, data_name='fact')

                    result = model(fact, self.config, self.gpu_list, acc_result=None, mode='serve')

                    # the size of charge_result = [number_of_class]
                    charge_result = torch.max(result['accuse'], 2)[1]
                    article_source_result = torch.max(result['article_source'], 2)[1]
                    article_result = torch.max(result['article'], 2)[1]
            
                    reply_text = ''

                    for key, value in charge_table.items():
                        if torch.equal(charge_result, value):
                            # reply_text += (f'The charge of this fact: {key}' + '\n')
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
                        reply_text = '查不到對應的資料，請檢查標點符號或以更完整的敘述再試一次！'
                    else:
                        self.client_socket.sendall(reply_text.encode())

            self.client_socket.close()
        else:
            print('Impossible output.')