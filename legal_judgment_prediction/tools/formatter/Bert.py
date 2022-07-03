import torch
import numpy as np

from transformers import BertTokenizer
# from pytorch_pretrained_bert.tokenization import BertTokenizer

from legal_judgment_prediction.tools.formatter.Basic import BasicFormatter


class BertLJP(BasicFormatter):
    def __init__(self, config, mode, *args, **kwargs):
        super().__init__(config, mode, *args, **kwargs)

        self.mode = mode
        self.max_len = config.getint('data', 'max_seq_length')
        self.tokenizer = BertTokenizer.from_pretrained(config.get('model', 'bert_path'))

        # Process charge
        self.charge2id = {}

        charge = open(config.get('data', 'charge_path'), 'r', encoding='UTF-8')

        for line in charge:
            self.charge2id[line.replace('\r', '').replace('\n', '')] = len(self.charge2id)

        # Process article_source
        self.article_source2id = {}

        article_source = open(config.get('data', 'article_source_path'), 'r', encoding='UTF-8')

        for line in article_source:
            self.article_source2id[line.replace('\r', '').replace('\n', '')] = len(self.article_source2id)

        # Process article
        self.article2id = {}

        article = open(config.get('data', 'article_path'), 'r', encoding='UTF-8')

        for line in article:
            self.article2id[line.replace('\r', '').replace('\n', '')] = len(self.article2id)


    def process(self, datas, *args, **kwargs):
        charge = []
        article_source = []
        article = []

        if self.mode == 'serve':
            if 'fact' in datas:
                one_fact = datas['fact']
                one_fact = self.tokenizer.tokenize(one_fact)

                while len(one_fact) < self.max_len:
                    one_fact.append('[PAD]')

                one_fact = one_fact[0:self.max_len]

                return torch.LongTensor(self.tokenizer.convert_tokens_to_ids(one_fact)).cuda()
            elif 'charge' in datas:
                charge_data = datas['charge']
                one_charge = np.zeros(len(self.charge2id), dtype=np.int)

                if self.charge2id.get(charge_data):
                    one_charge[self.charge2id[charge_data]] = 1
                else:
                    one_charge[self.charge2id['others']] = 1

                charge.append(one_charge.tolist())
                    
                return torch.LongTensor(charge).cuda()
            elif 'article_source' in datas:
                article_source_data = datas['article_source']
                one_article_source = np.zeros(len(self.article_source2id), dtype=np.int)

                if self.article_source2id.get(article_source_data):
                    one_article_source[self.article_source2id[article_source_data]] = 1
                else:
                    one_article_source[self.article_source2id['others']] = 1

                article_source.append(one_article_source.tolist())

                return torch.LongTensor(article_source).cuda()
            elif 'article' in datas:
                article_data = datas['article']
                one_article = np.zeros(len(self.article2id), dtype=np.int)

                if self.article2id.get(article_data):
                    one_article[self.article2id[article_data]] = 1
                else:
                    one_article[self.article2id['others']] = 1

                article.append(one_article.tolist())

                return torch.LongTensor(article).cuda()
        else:   # mode != 'serve'
            fact = []

            # Process fact
            for data in datas:
                one_fact = data['fact']
                one_fact = self.tokenizer.tokenize(one_fact)

                while len(one_fact) < self.max_len:
                    one_fact.append('[PAD]')

                one_fact = one_fact[0:self.max_len]
                fact.append(self.tokenizer.convert_tokens_to_ids(one_fact))

                # Process charge
                one_charge = np.zeros(len(self.charge2id), dtype=np.int)
                
                if self.charge2id.get(data['meta']['accusation']):
                    one_charge[self.charge2id[data['meta']['accusation']]] = 1
                else:
                    one_charge[self.charge2id['others']] = 1

                charge.append(one_charge.tolist())
                
                # Process article_source and article
                one_article_source = np.zeros(len(self.article_source2id), dtype=np.int)
                one_article = np.zeros(len(self.article2id), dtype=np.int)

                for item in data['meta']['relevant_articles']:
                    if self.article_source2id.get(item[0]):
                        one_article_source[self.article_source2id[item[0]]] = 1
                    else:
                        one_article_source[self.article_source2id['others']] = 1

                    if self.article2id.get(item[0] + item[1]):
                        one_article[self.article2id[item[0] + item[1]]] = 1
                    else:
                        one_article[self.article2id['others']] = 1

                article_source.append(one_article_source.tolist())
                article.append(one_article.tolist())
            
            fact = torch.LongTensor(fact)
            charge = torch.LongTensor(charge)
            article_source = torch.LongTensor(article_source)
            article = torch.LongTensor(article)

            return {'fact': fact, 'charge': charge, 'article_source': article_source, 'article': article}