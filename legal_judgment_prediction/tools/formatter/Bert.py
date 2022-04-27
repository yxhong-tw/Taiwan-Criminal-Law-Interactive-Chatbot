import torch
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer

from legal_judgment_prediction.tools.formatter.Basic import BasicFormatter


class BertLJP(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = BertTokenizer.from_pretrained(config.get('model', 'bert_path'))
        self.max_len = config.getint('data', 'max_seq_length')
        self.mode = mode

        # processing charge
        charge = open(config.get('data', 'charge_path'), 'r', encoding='utf-8')
        self.charge2id = {}

        for line in charge:
            self.charge2id[line.replace('\r', '').replace('\n', '')] = len(self.charge2id)

        # processing article_source
        article_source = open(config.get('data', 'article_source_path'), 'r', encoding='utf-8')
        self.article_source2id = {}

        for line in article_source:
            self.article_source2id[line.replace('\r', '').replace('\n', '')] = len(self.article_source2id)

        # processing article
        article = open(config.get('data', 'article_path'), 'r', encoding='utf-8')
        self.article2id = {}

        for line in article:
            self.article2id[line.replace('\r', '').replace('\n', '')] = len(self.article2id)


    def process(self, data, config, mode, *args, **params):
        input = []
        charge = []
        article_source = []
        article = []

        for temp in data:
            text = temp['fact']
            text = self.tokenizer.tokenize(text)

            while len(text) < self.max_len:
                text.append('[PAD]')

            text = text[0:self.max_len]
            input.append(self.tokenizer.convert_tokens_to_ids(text))

            # processing charge
            temp_charge = np.zeros(len(self.charge2id), dtype=np.int)
            
            if self.charge2id.get(temp['meta']['accusation']):
                temp_charge[self.charge2id[temp['meta']['accusation']]] = 1
            else:
                temp_charge[self.charge2id['others']] = 1

            charge.append(temp_charge.tolist())
            
            # processing article
            temp_article_source = np.zeros(len(self.article_source2id), dtype=np.int)
            temp_article = np.zeros(len(self.article2id), dtype=np.int)

            for name in temp['meta']['relevant_articles']:
                if self.article_source2id.get(name[0]):
                    temp_article_source[self.article_source2id[name[0]]] = 1
                else:
                    temp_article_source[self.article_source2id['others']] = 1

                if self.article2id.get(name[0] + name[1]):
                    temp_article[self.article2id[name[0] + name[1]]] = 1
                else:
                    temp_article[self.article2id['others']] = 1

            article_source.append(temp_article_source.tolist())
            article.append(temp_article.tolist())
        
        input = torch.LongTensor(input)
        charge = torch.LongTensor(charge)
        article_source = torch.LongTensor(article_source)
        article = torch.LongTensor(article)

        return {'text': input, 'accuse': charge, 'article_source': article_source, 'article': article}
