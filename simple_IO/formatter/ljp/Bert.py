import torch
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer

from simple_IO.formatter.Basic import BasicFormatter


class BertLJP(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "bert_path"))
        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode

        # processing charge
        charge = open(config.get("data", "charge_path"), "r", encoding="utf8")
        self.charge2id = {}

        for line in charge:
            self.charge2id[line.replace("\r", "").replace("\n", "")] = len(self.charge2id)

        # processing article_source
        article_source = open(config.get("data", "article_source_path"), "r", encoding="utf8")
        self.article_source2id = {}

        for line in article_source:
            self.article_source2id[line.replace("\r", "").replace("\n", "")] = len(self.article_source2id)
            
        # processing article
        article = open(config.get("data", "article_path"), "r", encoding="utf8")
        self.article2id = {}

        for line in article:
            self.article2id[line.replace("\r", "").replace("\n", "")] = len(self.article2id)

    def process(self, data, config, mode, *args, **params):
        charge = []
        article_source = []
        article = []

        if params['data_name'] == 'fact':
            data = self.tokenizer.tokenize(data)

            while len(data) < self.max_len:
                data.append("[PAD]")

            data = data[0:self.max_len]

            return torch.LongTensor(self.tokenizer.convert_tokens_to_ids(data)).cuda()
        elif params['data_name'] == 'charge':
            temp_charge = np.zeros(len(self.charge2id), dtype=np.int)

            if self.charge2id.get(data):
                temp_charge[self.charge2id[data]] = 1
            else:
                temp_charge[self.charge2id["others"]] = 1

            charge.append(temp_charge.tolist())
                
            return torch.LongTensor(charge).cuda()
        elif params['data_name'] == 'article_source':
            temp_article_source = np.zeros(len(self.article_source2id), dtype=np.int)

            if self.article_source2id.get(data):
                temp_article_source[self.article_source2id[data]] = 1
            else:
                temp_article_source[self.article_source2id['others']] = 1

            article_source.append(temp_article_source.tolist())

            return torch.LongTensor(article_source).cuda()
        else:   # params['data_name'] == 'article'
            temp_article = np.zeros(len(self.article2id), dtype=np.int)

            if self.article2id.get(data):
                temp_article[self.article2id[data]] = 1
            else:
                temp_article[self.article2id['others']] = 1

            article.append(temp_article.tolist())

            return torch.LongTensor(article).cuda()