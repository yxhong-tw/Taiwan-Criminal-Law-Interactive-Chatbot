import torch.nn as nn
import torch

from legal_judgment_prediction.model.bert_encoder import BertEncoder
from legal_judgment_prediction.model.bert_predictor import BertPredictor
from legal_judgment_prediction.utils import MultiLabelSoftmaxLoss
from legal_judgment_prediction.evaluation import multi_label_accuracy


class LJPBert(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(LJPBert, self).__init__()

        model_path = config.get('model', 'model_path')
        hidden_size = config.getint('model', 'hidden_size')
        articles_number = config.getint('data', 'articles_number')
        article_sources_number = config.getint('data', 'article_sources_number')
        accusations_number = config.getint('data', 'accusations_number')

        self.bert = BertEncoder(model_path=model_path)
        self.fc = BertPredictor(
            hidden_size=hidden_size
            , articles_number=articles_number
            , article_sources_number=article_sources_number
            , accusations_number=accusations_number)

        self.criterion = {
            'article': MultiLabelSoftmaxLoss(task_number=articles_number)
            , 'article_source': MultiLabelSoftmaxLoss(
                task_number=article_sources_number)
            , 'accusation': MultiLabelSoftmaxLoss(
                task_number=accusations_number)
        }
        self.accuracy_function = {
            'article': multi_label_accuracy,
            'article_source': multi_label_accuracy,
            'accusation': multi_label_accuracy
        }


    def initialize_multiple_gpus(self, gpus, *args, **kwargs):
        self.bert = nn.DataParallel(module=self.bert, device_ids=gpus)
        self.fc = nn.DataParallel(module=self.fc, device_ids=gpus)


    def forward(self, data, mode, acc_result):
        if mode == 'generate' or mode == 'serve':
            data = torch.unsqueeze(input=data, dim=0)
            output = self.bert(input=data)
            output = self.fc(tensor=output)

            return output
        else:
            fact = data['fact']
            output = self.bert(input=fact)
            output = self.fc(tensor=output)

            loss = 0
            for name in ['article', 'article_source', 'accusation']:
                loss += self.criterion[name](
                    outputs=output[name]
                    , labels=data[name])

            if acc_result is None:
                acc_result = {
                    'article': None
                    , 'article_source': None
                    , 'accusation': None
                }

            for name in ['article', 'article_source', 'accusation']:
                acc_result[name] = self.accuracy_function[name](
                    outputs=output[name]
                    , label=data[name]
                    , result=acc_result[name])

            return {'loss': loss, 'output': output, 'acc_result': acc_result}
