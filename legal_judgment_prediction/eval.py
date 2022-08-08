import logging
import torch
import gc

from timeit import default_timer as timer
from torch.autograd import Variable

from legal_judgment_prediction.utils import gen_time_str, output_value


logger = logging.getLogger(__name__)


def eval(parameters, config, gpu_list, *args, **kwargs):
    model_name = parameters['model_name']
    model = parameters['model']
    output_function = parameters['output_function']
    test_dataset = parameters['test_dataset']

    eval_one(
        model_name
        , model
        , test_dataset
        , 0, config
        , gpu_list
        , output_function
        , task='test'
    )


def eval_one(
        model_name
        , model
        , dataset
        , epoch
        , config
        , gpu_list
        , output_function
        , task
        , *args
        , **kwargs):
    model.eval()

    acc_result = None
    total_loss = 0
    total_len = len(dataset)
    start_time = timer()
    output_info = ''
    output_time = config.getint('output', 'output_time')
    step = -1

    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        results = model(config, data, 'eval', acc_result)

        if model_name == 'LJPBert':
            acc_result = results['acc_result']

        loss = results['loss']
        total_loss += float(loss)

        if step % output_time == 0:
            if model_name == 'LJPBart':
                output_info = output_function(config, total_loss, step)
            elif model_name == 'LJPBert':
                output_info = output_function(config, acc_result)

            delta_t = timer() - start_time

            # output_value(
            #     epoch
            #     , task
            #     , '%d/%d' % (step + 1, total_len)
            #     , '%s/%s' % (gen_time_str(delta_t)
            #     , gen_time_str(delta_t * (total_len - step - 1) / (step + 1)))
            #     , '%.3lf' % (total_loss / (step + 1))
            #     , output_info
            #     , '\r'
            #     , config
            # )

            output_value(
                epoch
                , task
                , f'{(step+1)}/{total_len}'
                , f'{gen_time_str(delta_t)}/\
{gen_time_str(delta_t*(total_len-step-1)/(step+1))}'
                , f'{(total_loss/(step+1))}'
                , output_info
                , '\r'
                , config
            )

    if step == -1:
        logger.error('There is no data in this dataset.')
        raise Exception('There is no data in this dataset.')

    if model_name == 'LJPBart':
        output_info = output_function(config, total_loss, step)
    elif model_name == 'LJPBert':
        output_info = output_function(config, acc_result)

    delta_t = timer() - start_time

    # output_value(epoch, task, '%d/%d' % (step + 1, total_len), '%s/%s' % (gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))), '%.3lf' % (total_loss / (step + 1)), output_info, None, config)

    output_value(
        epoch
        , task
        , f'{(step+1)}/{total_len}'
        , f'{gen_time_str(delta_t)}/\
{gen_time_str(delta_t*(total_len-step-1)/(step+1))}'
        , f'{(total_loss/(step+1))}'
        , output_info
        , None
        , config
    )

    if task == 'valid':
        model.train()

    gc.collect()
    torch.cuda.empty_cache()