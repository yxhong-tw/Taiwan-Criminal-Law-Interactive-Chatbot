import logging

from torch.utils.data import DataLoader

from tools.dataset.JsonFromFiles import JsonFromFilesDataset
from tools.formatter import init_formatter


logger = logging.getLogger(__name__)


def init_dataset(config, task, mode, *args, **kwargs):
    dataset_type_list = {
        'JsonFromFiles': JsonFromFilesDataset
    }

    try:
        dataset_type = config.get('data', '%s_dataset_type' % task)
    except Exception:
        logger.error('%s_dataset_type has not been defined in config file.' % task)
        raise AttributeError

    collate_fn = init_formatter(config, task, *args, **kwargs)

    if dataset_type in dataset_type_list:
        dataset = dataset_type_list[dataset_type](config, task, *args, **kwargs)

        try:
            batch_size = config.getint(mode, 'batch_size')
        except Exception:
            logger.error('batch_size has not been defined in [%s] in config file.' % mode)
            raise AttributeError

        try:
            shuffle = config.getboolean(mode, 'shuffle')
        except Exception:
            logger.error('shuffle has not been defined in [%s] in config file.' % mode)
            raise AttributeError

        try:
            reader_num = config.getint(mode, 'reader_num')
        except Exception:
            logger.error('reader has not been defined in [%s] in config file.' % mode)
            raise AttributeError

        drop_last = True

        if task == 'test':
            drop_last = False

        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=reader_num,
                                collate_fn=collate_fn,
                                drop_last=drop_last)

        return dataloader
    else:
        logger.error('There is no dataset_type called %s.' % dataset_type)
        raise AttributeError