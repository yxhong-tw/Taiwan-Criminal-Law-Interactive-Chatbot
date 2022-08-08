import logging
import torch.nn as nn
import torch
import numpy as np


logger = logging.getLogger(__name__)


# def get_tables(config, formatter, *args, **kwargs):
#     logger.info('Start to get tables.')

#     article_list, article_source_list, accusation_list = [], [], []

#     with open(
#             file=config.get('data', 'articles_path')
#             , mode='r'
#             , encoding='UTF-8') as file:
#         lines = file.readlines()

#         for index in range(len(lines)):
#             if lines[index][-1] == '\n':
#                 article_list.append(lines[index][0:-1])
#             else:
#                 article_list.append(lines[index])

#         file.close()

#     with open(
#             file=config.get('data', 'article_sources_path')
#             , mode='r'
#             , encoding='UTF-8') as file:
#         lines = file.readlines()

#         for index in range(len(lines)):
#             if lines[index][-1] == '\n':
#                 article_source_list.append(lines[index][0:-1])
#             else:
#                 article_source_list.append(lines[index])

#         file.close()
    
#     with open(
#             file=config.get('data', 'accusations_path')
#             , mode='r'
#             , encoding='UTF-8') as file:
#         lines = file.readlines()

#         for index in range(len(lines)):
#             if lines[index][-1] == '\n':
#                 accusation_list.append(lines[index][0:-1])
#             else:
#                 accusation_list.append(lines[index])

#         file.close()

#     article_table, article_source_table, accusation_table = {}, {}, {}
    
#     for data in article_list:
#         article_table[data] = formatter({'article': data})

#     for data in article_source_list:
#         article_source_table[data] = formatter({'article_source': data})

#     for data in accusation_list:
#         accusation_table[data] = formatter({'accusation': data})

#     logger.info('Get tables successfully.')

#     return article_table, article_source_table, accusation_table


# def string_process(
#         data
#         , preprocess=False
#         , converter=None):
#     if preprocess == True:
#         data = data.replace(' ', '').replace(',', '，')

#     if converter is not None:
#         data = converter.convert(data)

#     return data


def gen_time_str(t):
    t = int(t)
    minute = t // 60
    second = t % 60

    return '%2d:%02d' % (minute, second)


def output_value(epoch, mode, step, time, loss, info, end, config):
    try:
        delimiter = config.get('output', 'delimiter')
    except Exception:
        delimiter = ' '

    s = ''
    s = s + str(epoch) + ' '

    while len(s) < 7:
        s += ' '

    s = s + str(mode) + ' '

    while len(s) < 14:
        s += ' '

    s = s + str(step) + ' '

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
        information = s + end
        logger.info(information)
    else:
        information = s
        logger.info(information)


class MultiLabelSoftmaxLoss(nn.Module):
    def __init__(self, config, task_num=0):
        super(MultiLabelSoftmaxLoss, self).__init__()
        
        self.task_num = task_num
        self.criterion = []

        for a in range(0, self.task_num):
            try:
                ratio = config.getfloat('train', 'loss_weight_%d' % a)
                self.criterion.append(nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, ratio], dtype=np.float32)).cuda()))
            except Exception:
                self.criterion.append(nn.CrossEntropyLoss())


    def forward(self, outputs, labels):
        loss = 0
        
        for a in range(0, len(outputs[0])):
            o = outputs[:, a, :].view(outputs.size()[0], -1)
            loss += self.criterion[a](o, labels[:, a])

        return loss