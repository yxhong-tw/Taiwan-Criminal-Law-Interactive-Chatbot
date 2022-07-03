import random
import os
import json

from torch.utils.data import Dataset


class JsonFromFiles(Dataset):
    def __init__(self, config, task, encoding='UTF-8', *args, **kwargs):
        self.config = config
        self.file = os.path.join(config.get('data', f'{task}_file_path'), config.get('data', f'{task}_file_name'))
        self.data = []

        with open(self.file, 'r', encoding=encoding) as file:
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
