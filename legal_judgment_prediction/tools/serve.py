import torch
import socket

from legal_judgment_prediction.tools.formatter.Bert import BertLJP


def serve_simple_IO(parameters, config, gpu_list):
    model = parameters['model']
    model.eval()

    result = []
    acc_result = None

    charge_table, article_source_table, article_table = get_table(config, 'serve')

    while True:
        # === Get the fact (input) ===
        # Add the codes that can get message from Line-bot here
        # ============================
        fact = input('Enter a fact: ')

        if fact == 'exit':
            break

        fact = encode_data(config, 'serve', fact, 'fact')

        result = model(fact, config, gpu_list, acc_result, 'serve')

        # the size of charge_result = [number_of_class]
        charge_result = torch.max(result['accuse'], 2)[1]

        # TODO: the size of article_source_result = []
        article_source_result = torch.max(result['article_source'], 2)[1]

        # TODO: the size of article_result = []
        article_result = torch.max(result['article'], 2)[1]

        # === Get the accuse, article_source, article (output) ===
        # Add codes that can return the output back to Line-bot
        # ========================================================
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
    acc_result = None

    charge_table, article_source_table, article_table = get_table(config, 'serve')

    # socket connection
    
    model_HOST, model_PORT = '172.17.0.3', 8000
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((model_HOST, model_PORT))
    server.listen(10)

    while True:
        conn, addr = server.accept()
        clientMessage = str(conn.recv(1024), encoding='utf-8')
        print(f'connected, clientMessage:{clientMessage}')
        # === Get the fact (input) ===
        fact = clientMessage

        fact = encode_data(config, 'serve', fact, 'fact')

        result = model(fact, config, gpu_list, acc_result, 'serve')

        # the size of charge_result = [number_of_class]
        charge_result = torch.max(result['accuse'], 2)[1]

        # TODO: the size of article_source_result = []
        article_source_result = torch.max(result['article_source'], 2)[1]

        # TODO: the size of article_result = []
        article_result = torch.max(result['article'], 2)[1]

        # === Get the accuse, article_source, article (output) ===
        # Add codes that can return the output back to Line-bot
        # ========================================================
        
        reply_text = ''
        for key, value in charge_table.items():
            if torch.equal(charge_result, value):
                reply_text += f'The charge of this fact: {key}\n'
                # print(f'The charge of this fact: {key}')
                break

        for key, value in article_source_table.items():
            if torch.equal(article_source_result, value):
                reply_text += f'The article_source of this fact: {key}\n'
                # print(f'The article_source of this fact: {key}')
                break

        for key, value in article_table.items():
            if torch.equal(article_result, value):
                reply_text += f'The article of this fact: {key}'
                # print(f'The article of this fact: {key}')
                break
        
        serverMessage = reply_text
        conn.sendall(serverMessage.encode())
        conn.close()


def encode_data(config, mode, data, data_name, *args, **params):
    formatter = BertLJP(config, mode, *args, **params)

    return formatter.process(data, config, mode, data_name=data_name)


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
        charge_table[data] = encode_data(config, mode, data, 'charge')

    for data in article_source_list:
        article_source_table[data] = encode_data(config, mode, data, 'article_source')
    
    for data in article_list:
        article_table[data] = encode_data(config, mode, data, 'article')

    return charge_table, article_source_table, article_table