import os
import json
from torch.utils.data import Dataset
import random


class JsonFromFilesDataset(Dataset):
    def __init__(self, config, task, encoding='utf8', *args, **params):
        self.config = config
        self.task = task
        self.encoding = encoding
        self.file = os.path.join(config.get('data', '%s_file_path' % task), config.get('data', '%s_file_name' % task))

        self.data = []

        with open(self.file, 'r', encoding=self.encoding) as file:
            for line in file:
                self.data.append(json.loads(line))

        self.reduce = config.getboolean('data', 'reduce')

        if task == 'train':
            random.shuffle(self.data)
        else:
            self.reduce = False

        if self.reduce:
            self.reduce_ratio = config.getfloat('data', 'reduce_ratio')


    def __getitem__(self, item):
        if self.reduce:
            return self.data[random.randint(0, len(self.data) - 1)]

        return self.data[item]


    def __len__(self):
        if self.reduce:
            return int(self.reduce_ratio * len(self.data))

        return len(self.data)
