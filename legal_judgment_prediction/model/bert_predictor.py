import torch.nn as nn


class BertPredictor(nn.Module):
    def __init__(
            self
            , hidden_size
            , articles_number
            , article_sources_number
            , accusations_number
            , *args
            , **kwargs):
        super(BertPredictor, self).__init__()

        self.article_fc = nn.Linear(
            in_features=hidden_size
            , out_features=articles_number*2)
        self.article_source_fc = nn.Linear(
            in_features=hidden_size
            , out_features=article_sources_number*2)
        self.accusation_fc = nn.Linear(
            in_features=hidden_size
            , out_features=accusations_number*2)


    def forward(self, tensor):
        article = self.article_fc(input=tensor)
        article_source = self.article_source_fc(input=tensor)
        accusation = self.accusation_fc(input=tensor)
        
        batch = tensor.size()[0]
        article = article.view(batch, -1, 2)
        article_source = article_source.view(batch, -1, 2)
        accusation = accusation.view(batch, -1, 2)

        return {
            'article': article
            , 'article_source': article_source
            , 'accusation': accusation
        }
