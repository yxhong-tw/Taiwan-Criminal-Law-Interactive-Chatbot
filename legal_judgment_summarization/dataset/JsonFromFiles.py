import random
import os
import json

from torch.utils.data import Dataset


class JsonFromFiles(Dataset):
    def __init__(self, config, task, encoding='UTF-8', *args, **kwargs):
        self.file = os.path.join(
            config.get('data', f'{task}_file_path')
            , config.get('data', f'{task}_file_name'))
        self.data = []

        with open(file=self.file, mode='r', encoding=encoding) as json_file:
            for line in json_file:
                one_data = json.loads(line)
                self.data.append(one_data)

        self.reduce = config.getboolean('data', 'reduce')

        if task == 'train':
            random.shuffle(self.data)
        else:
            self.reduce = False

        if self.reduce == True:
            self.reduce_ratio = config.getfloat('data', 'reduce_ratio')


    def __getitem__(self, item, *args, **kwargs):
        if self.reduce == True:
            return self.data[random.randint(0, len(self.data)-1)]

        return self.data[item]


    def __len__(self, *args, **kwargs):
        if self.reduce == True:
            return int(self.reduce_ratio*len(self.data))

        return len(self.data)
