import torch

from transformers import BertTokenizer

from legal_judgment_prediction.formatter.utils import set_special_tokens


class BartFormatter:
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

        self.mode = mode
        self.tokenizer = \
            BertTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_path)
        self.add_tokens_at_beginning = add_tokens_at_beginning
        self.max_len = max_len


    def process(self, data, *args, **kwargs):
        if self.mode == 'serve':
            data = self.tokenizer.tokenize(data)
            data = set_special_tokens(
                add_tokens_at_beginning=self.add_tokens_at_beginning
                , max_len=self.max_len
                , data=data)

            result = torch.LongTensor(
                self.tokenizer.convert_tokens_to_ids(data))

            return result.cuda()
        else:
            facts = []
            articles = []
            article_sources = []
            accusations = []

            for one_data in data:
                fact = self.tokenizer.tokenize(one_data['fact'])
                fact = set_special_tokens(
                    add_tokens_at_beginning=self.add_tokens_at_beginning
                    , max_len=self.max_len
                    , data=fact)

                facts.append(self.tokenizer.convert_tokens_to_ids(fact))
                
                for relevant_article in one_data['meta']['relevant_articles']:
                    article = relevant_article[0] + relevant_article[1]
                    article = self.tokenizer.tokenize(article)
                    article = set_special_tokens(
                        add_tokens_at_beginning=self.add_tokens_at_beginning
                        , max_len=self.max_len
                        , data=article)

                    articles.append(
                        self.tokenizer.convert_tokens_to_ids(article))

                    article_source = relevant_article[0]
                    article_source = self.tokenizer.tokenize(article_source)
                    article_source = set_special_tokens(
                        add_tokens_at_beginning=self.add_tokens_at_beginning
                        , max_len=self.max_len
                        , data=article_source)

                    article_sources.append(
                        self.tokenizer.convert_tokens_to_ids(article_source))

                accusation = one_data['meta']['accusation']
                accusation = self.tokenizer.tokenize(accusation)
                accusation = set_special_tokens(
                    add_tokens_at_beginning=self.add_tokens_at_beginning
                    , max_len=self.max_len
                    , data=accusation)

                accusations.append(
                    self.tokenizer.convert_tokens_to_ids(accusation))
            
            facts = torch.LongTensor(facts)
            articles = torch.LongTensor(articles)
            article_sources = torch.LongTensor(article_sources)
            accusations = torch.LongTensor(accusations)

            return {
                'fact': facts.cuda()
                , 'article': articles.cuda()
                , 'article_source': article_sources.cuda()
                , 'accusation': accusations.cuda()
            }