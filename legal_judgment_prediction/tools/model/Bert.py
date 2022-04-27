import torch.nn as nn

from tools.model.BertEncoder import BertEncoder
from tools.utils import MultiLabelSoftmaxLoss
from tools.model.Predictor import LJPPredictor
from tools.accuracy import multi_label_accuracy


class LJPBert(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(LJPBert, self).__init__()

        self.bert = BertEncoder(config, gpu_list, *args, **params)
        self.fc = LJPPredictor(config, gpu_list, *args, **params)

        self.criterion = {
            'accuse': MultiLabelSoftmaxLoss(config, 80),
            'article_source': MultiLabelSoftmaxLoss(config, 21),
            'article': MultiLabelSoftmaxLoss(config, 90),
        }
        self.accuracy_function = {
            'accuse': multi_label_accuracy,
            'article_source': multi_label_accuracy,
            'article': multi_label_accuracy
        }


    def init_multi_gpu(self, device, config, *args, **params):
        self.bert = nn.DataParallel(self.bert, device_ids=device)
        self.fc = nn.DataParallel(self.fc, device_ids=device)


    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['text']
        y = self.bert(x)
        result = self.fc(y)

        loss = 0
        for name in ['accuse', 'article_source', 'article']:
            loss += self.criterion[name](result[name], data[name])

        if acc_result is None:
            acc_result = {'accuse': None, 'article_source': None, 'article': None}

        for name in ['accuse', 'article_source', 'article']:
            acc_result[name] = self.accuracy_function[name](result[name], data[name], config, acc_result[name])

        return {'loss': loss, 'acc_result': acc_result, 'output': result}
