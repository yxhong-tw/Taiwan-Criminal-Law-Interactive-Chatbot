import logging

from torch.utils.data import DataLoader

from legal_judgment_prediction.dataset.JsonFromFiles import JsonFromFiles
from legal_judgment_prediction.formatter import initialize_formatter


logger = logging.getLogger(__name__)


def initialize_dataloader(config, task, mode, batch_size, *args, **kwargs):
    logger.info(f'Start to initialize {task} dataloader.')

    dataset_types = {
        'JsonFromFiles': JsonFromFiles
    }
    
    dataset_type = config.get('data', f'{task}_dataset_type')

    if dataset_type in dataset_types:
        dataset = dataset_types[dataset_type](config, task)
        
        batch_size = batch_size
        shuffle = config.getboolean(mode, 'shuffle')
        num_workers = config.getint(mode, 'num_workers')
        collate_fn = initialize_formatter(config, mode, task)
        drop_last = True
        
        if task == 'test':
            drop_last = False

        logger.info(f'Initialize {task} dataloader successfully.')

        return DataLoader(
            dataset=dataset
            , batch_size=batch_size
            , shuffle=shuffle
            , num_workers=num_workers
            , collate_fn=collate_fn
            , drop_last=drop_last)
    else:
        logger.error(f'There is no dataset_type called {dataset_type}.')
        raise Exception(f'There is no dataset_type called {dataset_type}.')