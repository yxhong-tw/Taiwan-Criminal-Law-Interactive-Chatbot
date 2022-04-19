from simple_IO.formatter.ljp.Bert import BertLJP


def init_formatter(config, mode, *args, **params):
    formatter = BertLJP(config, mode, *args, **params)

    return formatter