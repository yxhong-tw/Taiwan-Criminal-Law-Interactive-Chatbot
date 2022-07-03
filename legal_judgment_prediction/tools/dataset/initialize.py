import logging

from torch.utils.data import DataLoader

from legal_judgment_prediction.tools.dataset.JsonFromFiles import JsonFromFiles
from legal_judgment_prediction.tools.formatter import initialize_formatter


logger = logging.getLogger(__name__)


def initialize_dataloader(config, task, mode, *args, **kwargs):
    dataset_list = {
        'JsonFromFiles': JsonFromFiles
    }
    
    # dataset_type = config.get('data', '%s_dataset_type' % task)
    dataset_type = config.get('data', f'{task}_dataset_type')

    if dataset_type in dataset_list:
        dataset = dataset_list[dataset_type](config, task, *args, **kwargs)
        
        batch_size = config.getint(mode, 'batch_size')
        shuffle = config.getboolean(mode, 'shuffle')
        num_workers = config.getint(mode, 'num_workers')

        collate_fn = initialize_formatter(config, task, mode, *args, **kwargs)

        drop_last = True
        if task == 'test':
            drop_last = False

        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                collate_fn=collate_fn,
                                drop_last=drop_last)

        return dataloader
    else:
        # logger.error('There is no dataset_type called %s.' % dataset_type)
        logger.error(f'There is no dataset_type called {dataset_type}.')
        raise Exception(f'There is no dataset_type called {dataset_type}.')