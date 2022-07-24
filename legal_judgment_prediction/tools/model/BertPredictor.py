import torch.nn as nn


class BertPredictor(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(BertPredictor, self).__init__()

        self.hidden_size = config.getint('model', 'hidden_size')
        self.charge_fc = nn.Linear(self.hidden_size, 148 * 2)
        self.article_source_fc = nn.Linear(self.hidden_size, 21 * 2)
        self.article_fc = nn.Linear(self.hidden_size, 90 * 2)


    def initialize_multiple_gpus(self, device, *args, **kwargs):
        pass


    def forward(self, tensor):
        charge = self.charge_fc(tensor)
        article_source = self.article_source_fc(tensor)
        article = self.article_fc(tensor)
        
        batch = tensor.size()[0]
        charge = charge.view(batch, -1, 2)
        article_source = article_source.view(batch, -1, 2)
        article = article.view(batch, -1, 2)

        return {'charge': charge, 'article_source': article_source, 'article': article}
