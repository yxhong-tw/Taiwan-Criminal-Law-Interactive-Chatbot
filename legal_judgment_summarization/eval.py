import logging
import torch
import gc

from timeit import default_timer as timer
from torch.autograd import Variable

from utils import gen_time_str, output_value


logger = logging.getLogger(__name__)


def eval_one(
        gpu_list
        , model
        , epoch
        , dataset
        , output_function
        , output_time
        , mode
        , *args
        , **kwargs):
    model.eval()

    step = -1
    total_loss = 0
    output_time = output_time
    output_info = ''
    start_time = timer()
    total_len = len(dataset)

    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        results = model(data, mode='train')
        loss = results['loss']
        total_loss += float(loss)

        if step % output_time == 0:
            output_info = output_function(total_loss, step)
            delta_t = timer() - start_time

            output_value(
                epoch=epoch
                , mode=mode
                , step=f'{(step+1)}/{total_len}'
                , time=f'{gen_time_str(delta_t)}/\
{gen_time_str(delta_t*(total_len-step-1)/(step+1))}'
                , loss=f'{(total_loss/(step+1))}'
                , info=output_info
                , end='\r'
            )

    if step == -1:
        logger.error('There is no data in this dataset.')
        raise Exception('There is no data in this dataset.')

    output_info = output_function(total_loss, step)
    delta_t = timer() - start_time

    output_value(
        epoch=epoch
        , mode=mode
        , step=f'{(step+1)}/{total_len}'
        , time=f'{gen_time_str(delta_t)}/\
{gen_time_str(delta_t*(total_len-step-1)/(step+1))}'
        , loss=f'{(total_loss/(step+1))}'
        , info=output_info
    )

    if mode == 'train':
        model.train()

    gc.collect()
    torch.cuda.empty_cache()