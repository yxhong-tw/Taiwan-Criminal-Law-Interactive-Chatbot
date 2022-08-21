import torch.nn as nn
import torch

from transformers import BertTokenizer

from legal_judgment_prediction.model.bart_model import BartModel


class LJPBart(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(LJPBart, self).__init__()

        model_path = config.get('model', 'model_path')

        self.bart = BartModel(model_path=model_path)
        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path)


    def initialize_multiple_gpus(self, gpus, *args, **kwargs):
        self.bart = nn.DataParallel(module=self.bart, device_ids=gpus)


    def forward(self, data, mode, acc_result, *args, **kwargs):
        if mode == 'serve':
            # The size of data after unsqueeze is 
            # [batch_size, seq_len] (1, 512).
            data = torch.unsqueeze(input=data, dim=0)

            # According to https://stackoverflow.com/questions/50442000/dataparallel-object-has-no-attribute-init-hidden,
            # because of DataParallel, the function will in 'module' attribute.
            tensor = self.bart.module.generate(input=data)

            texts = self.tokenizer.batch_decode(
                sequences=tensor
                , skip_special_tokens=False
                , clean_up_tokenization_spaces=False)[0]

            return texts
        else:
            outputs = self.bart(input=data['fact'], label=data['article'])
            loss, tensor = outputs
            texts = self.tokenizer.batch_decode(
                sequences=tensor
                , skip_special_tokens=True
                , clean_up_tokenization_spaces=False)[0]

            return {'loss': loss, 'text': texts, 'acc_result': acc_result}