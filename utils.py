import logging


logger = logging.getLogger(__name__)


def string_process(
        data
        , adjust_special_chars=False
        , converter=None):
    if adjust_special_chars == True:
        data = data.replace(' ', '').replace(',', '，')

    if converter is not None:
        data = converter.convert(data)

    return data


def get_tables(config, formatter, *args, **kwargs):
    logger.info('Start to get tables.')

    article_list, article_source_list, accusation_list = [], [], []

    with open(
            file=config.get('data', 'articles_path')
            , mode='r'
            , encoding='UTF-8') as file:
        lines = file.readlines()

        for index in range(len(lines)):
            if lines[index][-1] == '\n':
                article_list.append(lines[index][0:-1])
            else:
                article_list.append(lines[index])

        file.close()

    with open(
            file=config.get('data', 'article_sources_path')
            , mode='r'
            , encoding='UTF-8') as file:
        lines = file.readlines()

        for index in range(len(lines)):
            if lines[index][-1] == '\n':
                article_source_list.append(lines[index][0:-1])
            else:
                article_source_list.append(lines[index])

        file.close()
    
    with open(
            file=config.get('data', 'accusations_path')
            , mode='r'
            , encoding='UTF-8') as file:
        lines = file.readlines()

        for index in range(len(lines)):
            if lines[index][-1] == '\n':
                accusation_list.append(lines[index][0:-1])
            else:
                accusation_list.append(lines[index])

        file.close()

    article_table, article_source_table, accusation_table = {}, {}, {}
    
    for data in article_list:
        article_table[data] = formatter({'article': data})

    for data in article_source_list:
        article_source_table[data] = formatter({'article_source': data})

    for data in accusation_list:
        accusation_table[data] = formatter({'accusation': data})

    logger.info('Get tables successfully.')

    return article_table, article_source_table, accusation_table


def gen_time_str(t):
    t = int(t)
    minute = t // 60
    second = t % 60

    return '%2d:%02d' % (minute, second)


def output_value(epoch, mode, step, time, loss, info, end=None, delimiter=' '):
    # try:
    #     delimiter = config.get('output', 'delimiter')
    # except Exception:
    #     delimiter = ' '

    s = (str(epoch) + ' ')

    while len(s) < 7:
        s += ' '

    s += (str(mode) + ' ')

    while len(s) < 14:
        s += ' '

    s += (str(step) + ' ')

    while len(s) < 25:
        s += ' '

    s += str(time)

    while len(s) < 40:
        s += ' '

    s += str(loss)

    while len(s) < 48:
        s += ' '

    s += str(info)
    s = s.replace(' ', delimiter)

    if not (end is None):
        logger.info(s + end)
    else:
        logger.info(s)