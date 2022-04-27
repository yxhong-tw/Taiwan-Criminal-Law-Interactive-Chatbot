import logging
import torch
import torch.nn as nn
import numpy as np


logger = logging.getLogger(__name__)


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
        logger.info(s + end)
    else:
        logger.info(s)
        print(s)


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