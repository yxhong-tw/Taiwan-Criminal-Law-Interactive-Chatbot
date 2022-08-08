import torch.nn as nn


class BertPredictor(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(BertPredictor, self).__init__()

        self.hidden_size = config.getint('model', 'hidden_size')
        self.article_fc = nn.Linear(self.hidden_size, 90*2)
        self.article_source_fc = nn.Linear(self.hidden_size, 21*2)
        self.accusation_fc = nn.Linear(self.hidden_size, 148*2)


    def initialize_multiple_gpus(self, device, *args, **kwargs):
        pass


    def forward(self, tensor):
        article = self.article_fc(tensor)
        article_source = self.article_source_fc(tensor)
        accusation = self.accusation_fc(tensor)
        
        batch = tensor.size()[0]
        article = article.view(batch, -1, 2)
        article_source = article_source.view(batch, -1, 2)
        accusation = accusation.view(batch, -1, 2)

        return {
            'article': article
            , 'article_source': article_source
            , 'accusation': accusation
        }
