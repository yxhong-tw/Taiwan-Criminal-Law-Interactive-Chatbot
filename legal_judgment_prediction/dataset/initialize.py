import logging

from torch.utils.data import DataLoader

from legal_judgment_prediction.dataset.JsonFromFiles import JsonFromFiles
from legal_judgment_prediction.formatter import initialize_formatter


logger = logging.getLogger(__name__)


def initialize_dataloader(config, task, mode, batch_size, *args, **kwargs):
    logger.info(f'Start to initialize {task} dataloader.')

    dataset_name = config.get('data', f'{task}_dataset_name')
    datasets = {'JsonFromFiles': JsonFromFiles}

    if dataset_name in datasets:
        dataset = datasets[dataset_name](config=config, task=task)
        batch_size = batch_size
        shuffle = config.getboolean(mode, 'shuffle')
        num_workers = config.getint(mode, 'num_workers')
        collate_fn = initialize_formatter(config=config, mode=mode, task=task)
        drop_last = True
        
        if task == 'test':
            drop_last = False

        dataloader = DataLoader(
            dataset=dataset
            , batch_size=batch_size
            , shuffle=shuffle
            , num_workers=num_workers
            , collate_fn=collate_fn
            , drop_last=drop_last)

        logger.info(f'Initialize {task} dataloader successfully.')

        return dataloader
    else:
        logger.error(f'There is no dataset_type called {dataset_name}.')
        raise Exception(f'There is no dataset_type called {dataset_name}.')