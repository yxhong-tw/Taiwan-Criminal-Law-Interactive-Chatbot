class BasicFormatter:
    def __init__(self, config, *args, **kwargs):
        self.config = config


    def process(self, data, *args, **kwargs):
        return data