import torch
import torch.nn as nn

from simple_IO.model.encoder.BertEncoder import BertEncoder
from simple_IO.model.ljp.Predictor import LJPPredictor


class LJPBert(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(LJPBert, self).__init__()
        self.bert = BertEncoder(config, gpu_list, *args, **params)
        self.fc = LJPPredictor(config, gpu_list, *args, **params)


    def init_multi_gpu(self, device, config, *args, **params):
        self.bert = nn.DataParallel(self.bert, device_ids=device)
        self.fc = nn.DataParallel(self.fc, device_ids=device)


    def forward(self, data, config, gpu_list, acc_result, mode):
        x = torch.unsqueeze(data, 0)
        y = self.bert(x)
        result = self.fc(y)

        return result