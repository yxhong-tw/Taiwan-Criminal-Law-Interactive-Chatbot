import torch

from transformers import BertTokenizer

from legal_judgment_prediction.formatter.basic import BasicFormatter


class BartFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **kwargs):
        super().__init__(config, mode, *args, **kwargs)

        self.tokenizer = \
            BertTokenizer.from_pretrained(config.get('model', 'bart_path'))
        self.add_special_tokens = \
            config.getboolean('data', 'add_special_tokens')
        self.max_len = config.getint('data', 'max_seq_length')
        self.mode = mode


    def process(self, data, *args, **kwargs):
        if self.mode == 'serve':
            fact = data['fact']
            fact = self.tokenizer.tokenize(fact)

            if self.add_special_tokens == True:
                fact.insert(0, '[CLS]')
                fact.append('[SEP]')
            
            if len(fact) > self.max_len:
                fact = fact[0:self.max_len-1]
                fact.append('[SEP]')
            else:
                while len(fact) < self.max_len:
                    fact.append('[PAD]')

            result = \
                torch.LongTensor(self.tokenizer.convert_tokens_to_ids(fact))

            return result.cuda()
        else:
            facts = []
            articles = []
            article_sources = []
            accusations = []

            for one_data in data:
                fact = self.tokenizer.tokenize(one_data['fact'])

                if self.add_special_tokens == True:
                    fact.insert(0, '[CLS]')
                    fact.append('[SEP]')
                
                if len(fact) > self.max_len:
                    fact = fact[0:self.max_len-1]
                    fact.append('[SEP]')
                else:
                    while len(fact) < self.max_len:
                        fact.append('[PAD]')

                facts.append(self.tokenizer.convert_tokens_to_ids(fact))
                
                for relevant_article in one_data['meta']['relevant_articles']:
                    article = relevant_article[0] + relevant_article[1]
                    article = self.tokenizer.tokenize(article)

                    if self.add_special_tokens == True:
                        article.insert(0, '[CLS]')
                        article.append('[SEP]')

                    if len(article) > self.max_len:
                        article = article[0:self.max_len-1]
                        article.append('[SEP]')
                    else:
                        while len(article) < self.max_len:
                            article.append('[PAD]')

                    articles.append(
                        self.tokenizer.convert_tokens_to_ids(article))

                    article_source = relevant_article[0]
                    article_source = self.tokenizer.tokenize(article_source)

                    if self.add_special_tokens == True:
                        article_source.insert(0, '[CLS]')
                        article_source.append('[SEP]')

                    if len(article_source) > self.max_len:
                        article_source = article_source[0:self.max_len-1]
                        article_source.append('[SEP]')
                    else:
                        while len(article_source) < self.max_len:
                            article_source.append('[PAD]')

                    article_sources.append(
                        self.tokenizer.convert_tokens_to_ids(article_source))

                accusation = one_data['meta']['accusation']
                accusation = self.tokenizer.tokenize(accusation)

                if self.add_special_tokens == True:
                    accusation.insert(0, '[CLS]')
                    accusation.append('[SEP]')

                if len(accusation) > self.max_len:
                    accusation = accusation[0:self.max_len-1]
                    accusation.append('[SEP]')
                else:
                    while len(accusation) < self.max_len:
                        accusation.append('[PAD]')

                accusations.append(
                    self.tokenizer.convert_tokens_to_ids(accusation))
            
            facts = torch.LongTensor(facts)
            articles = torch.LongTensor(articles)
            article_sources = torch.LongTensor(article_sources)
            accusations = torch.LongTensor(accusations)

            return {
                'fact': facts
                , 'article': articles
                , 'article_source': article_sources
                , 'accusation': accusations
            }