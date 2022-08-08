import torch.nn as nn

from transformers import BartForConditionalGeneration


class BartModel(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(BartModel, self).__init__()

        self.bart = BartForConditionalGeneration.from_pretrained(
            config.get('model', 'bart_path'))


    def forward(self, input, label, *args, **kwargs):
        loss = self.bart(input_ids=input, labels=label)['loss']
        tensor = self.bart.generate(input)

        return loss, tensor


    def generate(self, input, *args, **kwargs):
        output = self.bart.generate(input)

        return output