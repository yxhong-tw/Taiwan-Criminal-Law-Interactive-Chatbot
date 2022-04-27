from legal_judgment_prediction.tools.formatter.Bert import BertLJP


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