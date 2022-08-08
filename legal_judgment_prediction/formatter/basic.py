class BasicFormatter:
    def __init__(self, config, mode, *args, **kwargs):
        self.config = config
        self.mode = mode

    def process(self, data, *args, **kwargs):
        return data