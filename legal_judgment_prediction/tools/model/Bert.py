import torch.nn as nn
import torch

from legal_judgment_prediction.tools.model.BertEncoder import BertEncoder
from legal_judgment_prediction.tools.model.BertPredictor import BertPredictor
from legal_judgment_prediction.tools.utils import MultiLabelSoftmaxLoss
from legal_judgment_prediction.tools.accuracy import multi_label_accuracy


class LJPBert(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(LJPBert, self).__init__()

        self.bert = BertEncoder(config, *args, **kwargs)
        self.fc = BertPredictor(config, *args, **kwargs)

        self.criterion = {
            'charge': MultiLabelSoftmaxLoss(config, 80),
            'article_source': MultiLabelSoftmaxLoss(config, 21),
            'article': MultiLabelSoftmaxLoss(config, 90),
        }
        self.accuracy_function = {
            'charge': multi_label_accuracy,
            'article_source': multi_label_accuracy,
            'article': multi_label_accuracy
        }


    def initialize_multiple_gpus(self, device, *args, **kwargs):
        self.bert = nn.DataParallel(self.bert, device_ids=device)
        self.fc = nn.DataParallel(self.fc, device_ids=device)


    def forward(self, config, data, mode, acc_result):
        if mode == 'serve':
            data = torch.unsqueeze(data, 0)
            output = self.bert(data)
            output = self.fc(output)

            return output
        else:
            fact = data['fact']
            outputs = self.bert(fact)
            outputs = self.fc(outputs)

            loss = 0
            for name in ['charge', 'article_source', 'article']:
                loss += self.criterion[name](outputs[name], data[name])

            if acc_result is None:
                acc_result = {'charge': None, 'article_source': None, 'article': None}

            for name in ['charge', 'article_source', 'article']:
                acc_result[name] = self.accuracy_function[name](outputs[name], data[name], config, acc_result[name])

            return {'loss': loss, 'acc_result': acc_result, 'output': outputs}
