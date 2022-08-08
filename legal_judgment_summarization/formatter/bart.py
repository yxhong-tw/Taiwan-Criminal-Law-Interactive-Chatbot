import torch

from transformers import BertTokenizer

from legal_judgment_summarization.formatter.basic import BasicFormatter


class BartFormatter(BasicFormatter):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.tokenizer = \
            BertTokenizer.from_pretrained(config.get('model', 'bart_path'))
        self.add_special_tokens = \
            config.getboolean('data', 'add_special_tokens')
        self.max_len = config.getint('data', 'max_seq_length')


    def process(self, data, *args, **kwargs):
        if type(data) == list:
            texts = []
            summaries = []

            for one_data in data:
                text = self.tokenizer.tokenize(one_data['text'])

                if self.add_special_tokens == True:
                    text.insert(0, '[CLS]')
                    text.append('[SEP]')
                
                if len(text) > self.max_len:
                    text = text[0:self.max_len-1]
                    text.append('[SEP]')
                else:
                    while len(text) < self.max_len:
                        text.append('[PAD]')

                texts.append(self.tokenizer.convert_tokens_to_ids(text))

                summary = \
                    self.tokenizer.tokenize(one_data['summary'])

                if self.add_special_tokens == True:
                    summary.insert(0, '[CLS]')
                    summary.append('[SEP]')
                
                if len(summary) > self.max_len:
                    summary = summary[0:self.max_len-1]
                    summary.append('[SEP]')
                else:
                    while len(summary) < self.max_len:
                        summary.append('[PAD]')

                summaries.append(self.tokenizer.convert_tokens_to_ids(summary))
            
            texts = torch.LongTensor(texts)
            summaries = torch.LongTensor(summaries)

            return {'text': texts, 'summary': summaries}
        elif type(data) == str:
            text = self.tokenizer.tokenize(data)

            if self.add_special_tokens == True:
                text.insert(0, '[CLS]')
                text.append('[SEP]')
            
            if len(text) > self.max_len:
                text = text[0:self.max_len-1]
                text.append('[SEP]')
            else:
                while len(text) < self.max_len:
                    text.append('[PAD]')

            text = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(text))

            # The size of data after unsqueeze
            # = [batch_size, seq_len]
            # = [1, 512].
            text = torch.unsqueeze(text, 0)

            return text.cuda()