import torch.nn as nn

# from transformers import BertModel
from pytorch_pretrained_bert import BertModel


class BertEncoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(BertEncoder, self).__init__()

        self.bert = BertModel.from_pretrained(config.get('model', 'bert_path'))


    def forward(self, input):
        _, output = self.bert(input, output_all_encoded_layers=False)

        return output
