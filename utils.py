import logging
import re


logger = logging.getLogger(__name__)


def string_process(
        data
        , adjust_special_chars=False
        , process_fact=False
        , converter=None):
    if adjust_special_chars == True:
        data = data.replace(' ', '')
        data = data.replace('\\', '')
        data = data.replace('`', '')
        data = data.replace('#', '')
        data = data.replace(',', '，')
        data = data.replace('：', ':')
        data = data.replace('；', ';')
        data = data.replace('？', '?')
        data = data.replace('！', '!')
        data = data.replace('（', '(')
        data = data.replace('）', ')')

    if process_fact == True:
        data = get_fact(string=data)

    if converter is not None:
        data = converter.convert(data)

    return data


def get_fact(string):
    current_index = 0
    chinese_number = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
    paragraph_list = []

    for index in range(len(string)):
        if index + 1 < len(string):
            if string[index] == '。' and string[index + 1] in chinese_number:
                for temp_index in range(index + 1, len(string)):
                    if string[temp_index] == '、':
                        paragraph_list.append(string[current_index:index + 1])
                        current_index = index + 1

                    if string[temp_index] not in chinese_number:
                        break
        else:
            paragraph_list.append(string[current_index:])

    paragraph_list = paragraph_list[:-1]

    for index in range(len(paragraph_list)):
        for temp_index in range(len(paragraph_list[index])):
            if paragraph_list[index][temp_index] == '、':
                paragraph_list[index] = paragraph_list[index][temp_index + 1:]
                break

    sentence_list = []
    for paragraph in paragraph_list:
        small_sentence_list = re.split(pattern=r'([，。])', string=paragraph)

        for sentence in small_sentence_list:
            sentence_list.append(sentence)

    for index, sentence in enumerate(iterable=sentence_list):
        if re.match(pattern=r'讵[^，]+悔', string=sentence) != None:
            sentence_list = sentence_list[index:]

    fact = ''
    for sentence in sentence_list:
        fact += sentence

    return fact


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


def log_results(epoch, stage, step, time, loss, info, end=None, delimiter=' '):
    s = (str(epoch) + ' ')

    while len(s) < 7:
        s += ' '

    s += (str(stage) + ' ')

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