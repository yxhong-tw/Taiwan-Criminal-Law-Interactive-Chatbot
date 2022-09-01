import torch.nn as nn

from transformers import BartForConditionalGeneration


class BartModel(nn.Module):
    def __init__(self, model_path, *args, **kwargs):
        super(BartModel, self).__init__()

        self.bart = BartForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_path)


    def forward(self, input, label, *args, **kwargs):
        loss = self.bart(input_ids=input, labels=label)['loss']
        tensor = self.bart.generate(input=input)

        return loss, tensor


    def generate(self, input, *args, **kwargs):
        output = self.bart.generate(inputs=input)

        return output