import torch
import numpy as np

from transformers import BertTokenizer

from legal_judgment_prediction.formatter.basic import BasicFormatter


class BertFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **kwargs):
        super().__init__(config, mode, *args, **kwargs)

        self.tokenizer = \
            BertTokenizer.from_pretrained(config.get('model', 'bert_path'))
        self.max_len = config.getint('data', 'max_seq_length')
        self.mode = mode

        self.article2id = {}

        articles = open(
            file=config.get('data', 'articles_path')
            , mode='r'
            , encoding='UTF-8')

        for article in articles:
            article = article.replace('\r', '').replace('\n', '')
            self.article2id[article] = len(self.article2id)

        self.article_source2id = {}

        article_sources = open(
            file=config.get('data', 'article_sources_path')
            , mode='r'
            , encoding='UTF-8')

        for article_source in article_sources:
            article_source = article_source.replace('\r', '').replace('\n', '')
            self.article_source2id[article_source] = len(self.article_source2id)

        self.accusation2id = {}

        accusations = open(
            file=config.get('data', 'accusations_path')
            , mode='r'
            , encoding='UTF-8')

        for accusation in accusations:
            accusation = accusation.replace('\r', '').replace('\n', '')
            self.accusation2id[accusation] = len(self.accusation2id)


    def process(self, data, *args, **kwargs):
        if self.mode == 'generate' or self.mode == 'serve':
            if 'article' in data and type(data) == dict:
                article = data['article']
                article_vector = np.zeros(len(self.article2id), dtype=np.int)

                if self.article2id.get(article):
                    article_vector[self.article2id[article]] = 1
                else:
                    article_vector[self.article2id['others']] = 1

                result = torch.LongTensor([article_vector.tolist()])
            elif 'article_source' in data and type(data) == dict:
                article_source = data['article_source']
                article_source_vector = \
                    np.zeros(len(self.article_source2id), dtype=np.int)

                if self.article_source2id.get(article_source):
                    article_source_vector[
                        self.article_source2id[article_source]] = 1
                else:
                    article_source_vector[self.article_source2id['others']] = 1

                result = torch.LongTensor([article_source_vector.tolist()])
            elif 'accusation' in data and type(data) == dict:
                accusation = data['accusation']
                accusation_vector = \
                    np.zeros(len(self.accusation2id), dtype=np.int)

                if self.accusation2id.get(accusation):
                    accusation_vector[self.accusation2id[accusation]] = 1
                else:
                    accusation_vector[self.accusation2id['others']] = 1

                result = torch.LongTensor([accusation_vector.tolist()])
            else:
                data = self.tokenizer.tokenize(data)

                while len(data) < self.max_len:
                    data.append('[PAD]')

                data = data[0:self.max_len]

                result = \
                    torch.LongTensor(self.tokenizer.convert_tokens_to_ids(data))

            return result.cuda()
        elif self.mode == 'serve':
            if 'fact' in data:
                fact = self.tokenizer.tokenize(data['fact'])

                while len(fact) < self.max_len:
                    fact.append('[PAD]')

                fact = fact[0:self.max_len]

                result = \
                    torch.LongTensor(self.tokenizer.convert_tokens_to_ids(fact))
            elif 'article' in data:
                article = data['article']
                article_vector = np.zeros(len(self.article2id), dtype=np.int)

                if self.article2id.get(article):
                    article_vector[self.article2id[article]] = 1
                else:
                    article_vector[self.article2id['others']] = 1

                result = torch.LongTensor([article_vector.tolist()])
            elif 'article_source' in data:
                article_source = data['article_source']
                article_source_vector = \
                    np.zeros(len(self.article_source2id), dtype=np.int)

                if self.article_source2id.get(article_source):
                    article_source_vector[
                        self.article_source2id[article_source]] = 1
                else:
                    article_source_vector[self.article_source2id['others']] = 1

                result = torch.LongTensor([article_source_vector.tolist()])
            elif 'accusation' in data:
                accusation = data['accusation']
                accusation_vector = \
                    np.zeros(len(self.accusation2id), dtype=np.int)

                if self.accusation2id.get(accusation):
                    accusation_vector[self.accusation2id[accusation]] = 1
                else:
                    accusation_vector[self.accusation2id['others']] = 1

                result = torch.LongTensor([accusation_vector.tolist()])

            return result.cuda()
        else:
            fact_vectors = []
            article_vectors = []
            article_source_vectors = []
            accusation_vectors = []

            for one_data in data:
                fact_vector = self.tokenizer.tokenize(one_data['fact'])

                while len(fact_vector) < self.max_len:
                    fact_vector.append('[PAD]')

                fact_vector = fact_vector[0:self.max_len]
                fact_vectors.append(
                    self.tokenizer.convert_tokens_to_ids(fact_vector))

                article_vector = np.zeros(len(self.article2id), dtype=np.int)
                article_source_vector = \
                    np.zeros(len(self.article_source2id), dtype=np.int)

                for relevant_article in one_data['meta']['relevant_articles']:
                    article = relevant_article[0] + relevant_article[1]
                    article_source = relevant_article[0]

                    if self.article2id.get(article):
                        article_vector[self.article2id[article]] = 1
                    else:
                        article_vector[self.article2id['others']] = 1

                    if self.article_source2id.get(article_source):
                        article_source_vector[
                            self.article_source2id[article_source]] = 1
                    else:
                        article_source_vector[
                            self.article_source2id['others']] = 1

                article_vectors.append(article_vector.tolist())
                article_source_vectors.append(article_source_vector.tolist())

                accusation_vector = \
                    np.zeros(len(self.accusation2id), dtype=np.int)
                
                if self.accusation2id.get(one_data['meta']['accusation']):
                    accusation_vector[
                        self.accusation2id[one_data['meta']['accusation']]] = 1
                else:
                    accusation_vector[self.accusation2id['others']] = 1

                accusation_vectors.append(accusation_vector.tolist())

            fact_vectors = torch.LongTensor(fact_vectors)
            article_vectors = torch.LongTensor(article_vectors)
            article_source_vectors = torch.LongTensor(article_source_vectors)
            accusation_vectors = torch.LongTensor(accusation_vectors)

            return {
                'fact': fact_vectors
                , 'article': article_vectors
                , 'article_source': article_source_vectors
                , 'accusation': accusation_vectors
            }