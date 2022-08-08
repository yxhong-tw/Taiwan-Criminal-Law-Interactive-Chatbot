import torch.nn as nn
import torch

from legal_judgment_prediction.model.bert_encoder import BertEncoder
from legal_judgment_prediction.model.bert_predictor import BertPredictor
from legal_judgment_prediction.utils import MultiLabelSoftmaxLoss
from legal_judgment_prediction.evaluation import multi_label_accuracy


class LJPBert(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(LJPBert, self).__init__()

        self.bert = BertEncoder(config)
        self.fc = BertPredictor(config)

        self.criterion = {
            'article': MultiLabelSoftmaxLoss(config, 90),
            'article_source': MultiLabelSoftmaxLoss(config, 21),
            'accusation': MultiLabelSoftmaxLoss(config, 148)
        }
        self.accuracy_function = {
            'article': multi_label_accuracy,
            'article_source': multi_label_accuracy,
            'accusation': multi_label_accuracy
        }


    def initialize_multiple_gpus(self, device, *args, **kwargs):
        self.bert = nn.DataParallel(self.bert, device_ids=device)
        self.fc = nn.DataParallel(self.fc, device_ids=device)


    def forward(self, config, data, mode, acc_result):
        if mode == 'generate' or mode == 'serve':
            data = torch.unsqueeze(data, 0)
            output = self.bert(data)
            output = self.fc(output)

            return output
        else:
            fact = data['fact']
            output = self.bert(fact)
            output = self.fc(output)

            loss = 0
            for name in ['article', 'article_source', 'accusation']:
                loss += self.criterion[name](output[name], data[name])

            if acc_result is None:
                acc_result = {
                    'article': None
                    , 'article_source': None
                    , 'accusation': None
                }

            for name in ['article', 'article_source', 'accusation']:
                acc_result[name] = self.accuracy_function[name](
                    output[name]
                    , data[name]
                    , config
                    , acc_result[name])

            return {'output': output, 'loss': loss, 'acc_result': acc_result}
