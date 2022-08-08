import logging
import os
import torch
import gc

from timeit import default_timer as timer
from torch.autograd import Variable

from utils import gen_time_str, output_value
from legal_judgment_summarization.eval import eval_one


logger = logging.getLogger(__name__)


def train(parameters, *args, **kwargs):
    model = parameters['model']
    optimizer_name = parameters['optimizer_name']
    optimizer = parameters['optimizer']
    exp_lr_scheduler = parameters['exp_lr_scheduler']
    trained_epoch = parameters['trained_epoch'] + 1
    # global_step = parameters['global_step']
    train_dataset = parameters['train_dataloader']
    valid_dataset = parameters['valid_dataloader']
    output_function = parameters['output_function']
    total_epoch = parameters['epoch']
    # step_size = parameters['step_size']
    # lr_multiplier = parameters['lr_multiplier']
    output_path = parameters['output_path']

    if not os.path.exists(output_path):
        logger.warn(
            f'The path of output {output_path} does not exist.')
        logger.info('Make the directory automatically.')

        os.makedirs(output_path)

    output_time = parameters['output_time']
    test_time = parameters['test_time']
    # exp_lr_scheduler = lr_scheduler.StepLR(
    #     optimizer=optimizer, step_size=step_size, gamma=lr_multiplier)
    # exp_lr_scheduler.step(trained_epoch)

    logger.info('Start to train model.')

    logger.info(
        'Epoch  Stage  Iterations  Time Usage    Loss    Output Information')

    total_len = len(train_dataset)

    for current_epoch in range(trained_epoch, total_epoch):
        # exp_lr_scheduler.step(current_epoch)

        for param_group in optimizer.param_groups:
            logger.info(f'Current learning rate: {param_group["lr"]}')

        start_time = timer()
        current_epoch = current_epoch
        total_loss = 0
        output_info = ""
        step = -1

        for step, data in enumerate(train_dataset):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(parameters['gpu_list']) > 0:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

            optimizer.zero_grad()

            results = model(data, mode='train')
            loss = results['loss']
            total_loss += float(loss)

            loss.backward()
            optimizer.step()

            if step % output_time == 0:
                output_info = output_function(total_loss, step)
                delta_t = timer() - start_time

                output_value(
                    epoch=current_epoch
                    , mode='train'
                    , step=f'{(step+1)}/{total_len}'
                    , time=f'{gen_time_str(delta_t)}/\
{gen_time_str(delta_t*(total_len-step-1)/(step+1))}'
                    , loss=f'{(total_loss/(step+1))}'
                    , info=output_info
                    , end='\r'
                )

            # global_step += 1

        exp_lr_scheduler.step()

        output_value(
            epoch=current_epoch
            , mode='train'
            , step=f'{(step+1)}/{total_len}'
            , time=f'{gen_time_str(delta_t)}/\
{gen_time_str(delta_t*(total_len-step-1)/(step+1))}'
            , loss=f'{(total_loss/(step+1))}'
            , info=output_info
        )

        if step == -1:
            logger.error('There is no data in this dataset.')
            raise Exception('There is no data in this dataset.')

        checkpoint(
            # , global_step=global_step
            model=model
            , optimizer_name=optimizer_name
            , optimizer=optimizer
            , trained_epoch=current_epoch
            , exp_lr_scheduler=exp_lr_scheduler
            , file=os.path.join(output_path, f'checkpoint_{current_epoch}.pkl')
        )

        gc.collect()
        torch.cuda.empty_cache()

        if current_epoch % test_time == 0:
            with torch.no_grad():
                eval_one(
                    gpu_list=parameters['gpu_list']
                    , model=model
                    , epoch=current_epoch
                    , dataset=valid_dataset
                    , output_function=output_function
                    , output_time=output_time
                    , mode='eval'
                )


def checkpoint(
        # , global_step
        model
        , optimizer_name
        , optimizer
        , exp_lr_scheduler
        , trained_epoch
        , file):
    model_to_save = model.module if hasattr(model, 'module') else model

    save_params = {
        'model': model_to_save.state_dict()
        , 'optimizer_name': optimizer_name
        , 'optimizer': optimizer.state_dict()
        , 'exp_lr_scheduler': exp_lr_scheduler.state_dict()
        , 'trained_epoch': trained_epoch
        # , 'global_step': global_step
    }

    try:
        torch.save(save_params, file)
    except Exception:
        logger.error(f'Failed to save model with error {Exception}.')
        raise Exception