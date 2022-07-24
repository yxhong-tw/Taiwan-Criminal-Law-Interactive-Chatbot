import torch
import torch.nn as nn

from transformers import BertTokenizer

from legal_judgment_prediction.tools.model.BartModel import BartModel
# from legal_judgment_prediction.tools.model.Predictor import LJPPredictor


class LJPBart(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(LJPBart, self).__init__()

        self.bart = BartModel(config, *args, **kwargs)
        # self.tokenizer = BertTokenizer.from_pretrained(config.get('model', 'bart_path'))
        self.tokenizer = BertTokenizer.from_pretrained(config.get('model', 'bart_path'))


    def initialize_multiple_gpus(self, device, *args, **kwargs):
        self.bart = nn.DataParallel(self.bart, device_ids=device)


    def forward(self, config, data, mode, acc_result):
        if mode == 'serve':
            # The size of data after unsqueeze = [batch_size, seq_len] = [1, 512]
            data = torch.unsqueeze(data, 0)

            # According to https://stackoverflow.com/questions/50442000/dataparallel-object-has-no-attribute-init-hidden
            # Because of DataParallel, the function will in 'module' attribute
            tensors = self.bart.module.generate(data)

            # TODO: Remove the space in texts            
            # texts = self.tokenizer.batch_decode(tensors, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            texts = self.tokenizer.batch_decode(tensors, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

            return texts
        else:   # mode == 'train', 'eval' or 'test'
            # data['text'] -> input, data['article'] -> label
            outputs = self.bart(data['fact'], data['article'])
            loss, tensors = outputs
            texts = self.tokenizer.batch_decode(tensors, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            return {'loss': loss, 'text': texts}