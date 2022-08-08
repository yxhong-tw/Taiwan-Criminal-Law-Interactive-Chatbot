import torch.nn as nn

from transformers import BartForConditionalGeneration


class BartModel(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(BartModel, self).__init__()

        self.bart = BartForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=config.get('model', 'bart_path'))


    def forward(self, input, label, *args, **kwargs):
        loss = self.bart(input_ids=input, labels=label)['loss']
        tensor = self.bart.generate(input, max_length=512)

        return loss, tensor


    def generate(self, input, *args, **kwargs):
        output = self.bart.generate(input, max_length=512)

        return output