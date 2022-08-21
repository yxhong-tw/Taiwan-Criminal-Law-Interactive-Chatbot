import numpy as np
import torch

from transformers import BertTokenizer

from legal_judgment_prediction.formatter.utils import set_special_tokens


class BertFormatter:
    def __init__(
            self
            , config
            , mode
            , *args
            , **kwargs):
        model_path = config.get('model', 'model_path')
        add_tokens_at_beginning = \
            config.getboolean('data', 'add_tokens_at_beginning')
        max_len = config.getint('data', 'max_len')
        articles_path = config.get('data', 'articles_path')
        article_sources_path = config.get('data', 'article_sources_path')
        accusations_path = config.get('data', 'accusations_path')

        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path)
        self.add_tokens_at_beginning = add_tokens_at_beginning
        self.max_len = max_len

        self.article2id = {}

        articles = open(
            file=articles_path
            , mode='r'
            , encoding='UTF-8')

        for article in articles:
            article = article.replace('\r', '').replace('\n', '')
            self.article2id[article] = len(self.article2id)

        self.article_source2id = {}

        article_sources = open(
            file=article_sources_path
            , mode='r'
            , encoding='UTF-8')

        for article_source in article_sources:
            article_source = article_source.replace('\r', '').replace('\n', '')
            self.article_source2id[article_source] = len(self.article_source2id)

        self.accusation2id = {}

        accusations = open(
            file=accusations_path
            , mode='r'
            , encoding='UTF-8')

        for accusation in accusations:
            accusation = accusation.replace('\r', '').replace('\n', '')
            self.accusation2id[accusation] = len(self.accusation2id)


    def process(self, data, *args, **kwargs):
        if self.mode == 'generate' or self.mode == 'serve':
            if isinstance(data, dict):
            # if type(data) == dict:
                if 'article' in data.keys():
                    article = data['article']
                    article_vector = np.zeros(
                        shape=len(self.article2id)
                        , dtype=np.int)

                    if self.article2id.get(article):
                        article_vector[self.article2id[article]] = 1
                    else:
                        article_vector[self.article2id['others']] = 1

                    result = torch.LongTensor([article_vector.tolist()])
                elif 'article_source' in data.keys():
                    article_source = data['article_source']
                    article_source_vector = np.zeros(
                        shape=len(self.article_source2id)
                        , dtype=np.int)

                    if self.article_source2id.get(article_source):
                        article_source_vector[
                            self.article_source2id[article_source]] = 1
                    else:
                        article_source_vector[
                            self.article_source2id['others']] = 1

                    result = torch.LongTensor([article_source_vector.tolist()])
                elif 'accusation' in data.keys():
                    accusation = data['accusation']
                    accusation_vector = np.zeros(
                        shape=len(self.accusation2id)
                        , dtype=np.int)

                    if self.accusation2id.get(accusation):
                        accusation_vector[self.accusation2id[accusation]] = 1
                    else:
                        accusation_vector[self.accusation2id['others']] = 1

                    result = torch.LongTensor([accusation_vector.tolist()])
            else:
                data = self.tokenizer.tokenize(data)
                data = set_special_tokens(
                    add_tokens_at_beginning=self.add_tokens_at_beginning
                    , max_len=self.max_len
                    , data=data)

                result = torch.LongTensor(
                    self.tokenizer.convert_tokens_to_ids(data))

            return result.cuda()
        else:
            fact_vectors = []
            article_vectors = []
            article_source_vectors = []
            accusation_vectors = []

            for one_data in data:
                fact = self.tokenizer.tokenize(one_data['fact'])
                fact = set_special_tokens(
                    add_tokens_at_beginning=self.add_tokens_at_beginning
                    , max_len=self.max_len
                    , data=fact)

                fact_vectors.append(
                    self.tokenizer.convert_tokens_to_ids(fact))

                article_vector = np.zeros(len(self.article2id), dtype=np.int)
                article_source_vector = np.zeros(
                    len(self.article_source2id)
                    , dtype=np.int)

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

                accusation_vector = np.zeros(
                    len(self.accusation2id)
                    , dtype=np.int)
                
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
                'fact': fact_vectors.cuda()
                , 'article': article_vectors.cuda()
                , 'article_source': article_source_vectors.cuda()
                , 'accusation': accusation_vectors.cuda()
            }