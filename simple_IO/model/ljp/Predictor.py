import torch.nn as nn


class LJPPredictor(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(LJPPredictor, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")
        self.charge_fc = nn.Linear(self.hidden_size, 80 * 2)
        self.article_source_fc = nn.Linear(self.hidden_size, 21 * 2)
        self.article_fc = nn.Linear(self.hidden_size, 90 * 2)

    def init_multi_gpu(self, device, config, *args, **params):
        pass

    def forward(self, h):
        charge = self.charge_fc(h)
        article_source = self.article_source_fc(h)
        article = self.article_fc(h)
        
        batch = h.size()[0]
        charge = charge.view(batch, -1, 2)
        article_source = article_source.view(batch, -1, 2)
        article = article.view(batch, -1, 2)

        return {"accuse": charge, "article": article, "article_source": article_source}
