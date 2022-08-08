import logging
import os
import torch
import gc

from torch.optim import lr_scheduler
from timeit import default_timer as timer
from torch.autograd import Variable

from legal_judgment_prediction.utils import gen_time_str, output_value
from legal_judgment_prediction.eval import eval_one


logger = logging.getLogger(__name__)


def train(parameters, config, gpu_list, do_test, *args, **kwargs):
    output_time = config.getint('output', 'output_time')
    test_time = config.getint('output', 'test_time')

    output_path = os.path.join(
        config.get('output', 'model_path')
        , config.get('output', 'model_name')
        , config.get('data', 'label')
        , config.get('data', 'range'))

    if os.path.exists(output_path):
        logger.warn('The output path already exists.')

    os.makedirs(output_path, exist_ok=True)

    model_name = parameters['model_name']
    model = parameters['model']
    global_step = parameters['global_step']
    output_function = parameters['output_function']
    epoch = config.getint('train', 'epoch')
    train_dataset = parameters['train_dataset']
    valid_dataset = parameters['valid_dataset']

    if do_test == True:
        test_dataset = parameters['test_dataset']

    optimizer = parameters['optimizer']
    step_size = config.getint('train', 'step_size')
    gamma = config.getfloat('train', 'lr_multiplier')
    exp_lr_scheduler = \
        lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    trained_epoch = parameters['trained_epoch'] + 1
    exp_lr_scheduler.step(trained_epoch)

    logger.info('Start to train model.')

    logger.info(
        'Epoch  Stage  Iterations  Time Usage    Loss    Output Information')

    total_len = len(train_dataset)

    for epoch_num in range(trained_epoch, epoch):
        start_time = timer()
        current_epoch = epoch_num

        exp_lr_scheduler.step(current_epoch)

        acc_result = None
        total_loss = 0

        output_info = ""
        step = -1

        for step, data in enumerate(train_dataset):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

            optimizer.zero_grad()

            results = model(config, data, 'train', acc_result)

            if model_name == 'LJPBert':
                acc_result = results['acc_result']

            loss = results['loss']
            total_loss += float(loss)

            loss.backward()
            optimizer.step()

            if step % output_time == 0:
                if model_name == 'LJPBart':
                    output_info = output_function(config, total_loss, step)
                elif model_name == 'LJPBert':
                    output_info = output_function(config, acc_result)

                delta_t = timer() - start_time

                # output_value(current_epoch, 'train', '%d/%d' % (step + 1, total_len), '%s/%s' % (gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))), '%.3lf' % (total_loss / (step + 1)), output_info, '\r', config)

                output_value(
                    current_epoch
                    , 'train'
                    , f'{(step+1)}/{total_len}'
                    , f'{gen_time_str(delta_t)}/\
{gen_time_str(delta_t*(total_len-step-1)/(step+1))}'
                    , f'{(total_loss/(step+1))}'
                    , output_info
                    , '\r'
                    , config
                )

            global_step += 1

        # output_value(current_epoch, 'train', '%d/%d' % (step + 1, total_len), '%s/%s' % (gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))), '%.3lf' % (total_loss / (step + 1)), output_info, None, config)

        output_value(
            current_epoch
            , 'train'
            , f'{(step+1)}/{total_len}'
            , f'{gen_time_str(delta_t)}/\
{gen_time_str(delta_t*(total_len-step-1)/(step+1))}'
            , f'{(total_loss/(step+1))}'
            , output_info
            , None
            , config
        )

        if step == -1:
            logger.error('There is no data in this dataset.')
            raise Exception('There is no data in this dataset.')

        checkpoint(
            os.path.join(output_path
            , f'checkpoint_{current_epoch}.pkl')
            , model
            , optimizer
            , current_epoch
            , config
            , global_step
        )

        gc.collect()
        torch.cuda.empty_cache()

        if current_epoch % test_time == 0:
            with torch.no_grad():
                eval_one(
                    model_name
                    , model
                    , valid_dataset
                    , current_epoch
                    , config
                    , gpu_list
                    , output_function
                    , task='valid'
                )

                if do_test:
                    eval_one(
                        model_name
                        , model
                        , test_dataset
                        , current_epoch
                        , config
                        , gpu_list
                        , output_function
                        , task='test'
                    )


def checkpoint(file, model, optimizer, trained_epoch, config, global_step):
    model_to_save = model.module if hasattr(model, 'module') else model

    save_params = {
        'model': model_to_save.state_dict(),
        'optimizer_name': config.get('train', 'optimizer'),
        'optimizer': optimizer.state_dict(),
        'trained_epoch': trained_epoch,
        'global_step': global_step
    }

    try:
        torch.save(save_params, file)
    except Exception:
        logger.error(f'Failed to save model with error {Exception}.')
        raise Exception