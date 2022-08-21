import torch.nn as nn

from pytorch_pretrained_bert import BertModel


class BertEncoder(nn.Module):
    def __init__(self, model_path, *args, **kwargs):
        super(BertEncoder, self).__init__()

        self.bert = BertModel.from_pretrained(
            pretrained_model_name_or_path=model_path)


    def forward(self, input):
        _, output = self.bert(input_ids=input, output_all_encoded_layers=False)

        return output
