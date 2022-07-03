import logging
import torch

from transformers import BertTokenizer

from legal_judgment_prediction.tools.formatter.Basic import BasicFormatter


logger = logging.getLogger(__name__)


class BartLJP(BasicFormatter):
    def __init__(self, config, mode, *args, **kwargs):
        super().__init__(config, mode, *args, **kwargs)

        self.mode = mode
        self.max_len = config.getint('data', 'max_seq_length')
        self.tokenizer = BertTokenizer.from_pretrained(config.get('model', 'bart_path'))


    def process(self, datas, *args, **kwargs):
        if self.mode == 'serve':
            if 'fact' in datas:
                one_fact = data['fact']    
                one_fact = self.tokenizer.tokenize(one_fact)
                
                while len(one_fact) < self.max_len:
                    one_fact.append('[PAD]')

                one_fact = one_fact[0:self.max_len]

                return torch.LongTensor(self.tokenizer.convert_tokens_to_ids(one_fact)).cuda()
            else:
                logger.error('The type of datas is invalid.')
                raise Exception('The type of datas is invalid.')
        else:
            fact = []
            charge = []
            article_source = []
            article = []

            for data in datas:
                # Process fact
                one_fact = data['fact']    
                one_fact = self.tokenizer.tokenize(one_fact)
                
                while len(one_fact) < self.max_len:
                    one_fact.append('[PAD]')

                one_fact = one_fact[0:self.max_len]

                fact.append(self.tokenizer.convert_tokens_to_ids(one_fact))

                # Process charge
                one_charge = data['meta']['accusation']
                one_charge = self.tokenizer.tokenize(one_charge)

                while len(one_charge) < self.max_len:
                    one_charge.append('[PAD]')

                one_charge = one_charge[0:self.max_len]

                charge.append(self.tokenizer.convert_tokens_to_ids(one_charge))
                
                for relevant_article in data['meta']['relevant_articles']:
                    # Process article_source
                    one_article_source = relevant_article[0]
                    one_article_source = self.tokenizer.tokenize(one_article_source)

                    while len(one_article_source) < self.max_len:
                        one_article_source.append('[PAD]')

                    one_article_source = one_article_source[0:self.max_len]

                    article_source.append(self.tokenizer.convert_tokens_to_ids(one_article_source))

                    # Process article
                    one_article = relevant_article[0] + relevant_article[1]
                    one_article = self.tokenizer.tokenize(one_article)

                    while len(one_article) < self.max_len:
                        one_article.append('[PAD]')

                    one_article = one_article[0:self.max_len]

                    article.append(self.tokenizer.convert_tokens_to_ids(one_article))
            
            fact = torch.LongTensor(fact)
            charge = torch.LongTensor(charge)
            article_source = torch.LongTensor(article_source)
            article = torch.LongTensor(article)

            return {'fact': fact, 'charge': charge, 'article_source': article_source, 'article': article}
