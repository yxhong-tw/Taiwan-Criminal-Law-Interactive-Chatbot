import configparser
import functools


class ConfigParser:
    def __init__(self, *args, **params):
        self.config = configparser.RawConfigParser(*args, **params)

    def read(self, filenames, encoding=None):
        self.config.read(filenames, encoding=encoding)


def _build_func(func_name):
    @functools.wraps(getattr(configparser.RawConfigParser, func_name))

    def func(self, *args, **kwargs):
        return getattr(self.config, func_name)(*args, **kwargs)

    return func


def create_config(path):
    for func_name in dir(configparser.RawConfigParser):
        if not func_name.startswith('_') and func_name != "read":
            setattr(ConfigParser, func_name, _build_func(func_name))

    config = ConfigParser()
    config.read(path)

    return config
