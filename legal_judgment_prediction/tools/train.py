import logging
import os
import shutil
import torch

from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from timeit import default_timer as timer
from torch.autograd import Variable

from legal_judgment_prediction.tools.initialize import init_dataset
from legal_judgment_prediction.tools.utils import gen_time_str, output_value
from legal_judgment_prediction.tools.eval import eval_one


logger = logging.getLogger(__name__)


def train(parameters, config, gpu_list, do_test=False):
    epoch = config.getint('train', 'epoch')

    output_time = config.getint('output', 'output_time')
    test_time = config.getint('output', 'test_time')

    output_path = os.path.join(config.get('output', 'model_path'), config.get('output', 'model_name'))

    if os.path.exists(output_path):
        logger.warning('Output path exists. Check whether need to change the name of model.')

    os.makedirs(output_path, exist_ok=True)

    trained_epoch = parameters['trained_epoch'] + 1
    model = parameters['model']
    optimizer = parameters['optimizer']
    global_step = parameters['global_step']
    output_function = parameters['output_function']
    train_dataset = parameters['train_dataset']
    valid_dataset = parameters['valid_dataset']

    if do_test:
        test_dataset = init_dataset(config, task='test', mode='eval')

    # tensorboardX
    # -----
    if trained_epoch == 0:
        shutil.rmtree(os.path.join(config.get('output', 'tensorboard_path'), config.get('output', 'model_name')), True)

    os.makedirs(os.path.join(config.get('output', 'tensorboard_path'), config.get('output', 'model_name')), exist_ok=True)

    writer = SummaryWriter(os.path.join(config.get('output', 'tensorboard_path'), config.get('output', 'model_name')), config.get('output', 'model_name'))
    # -----

    step_size = config.getint('train', 'step_size')
    gamma = config.getfloat('train', 'lr_multiplier')
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    exp_lr_scheduler.step(trained_epoch)

    logger.info('Training start....')

    logger.info('Epoch  Stage  Iterations  Time Usage    Loss    Output Information')

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

            results = model(data, config, gpu_list, acc_result, 'train')

            loss, acc_result = results['loss'], results['acc_result']
            total_loss += float(loss)

            loss.backward()
            optimizer.step()

            if step % output_time == 0:
                output_info = output_function(acc_result, config)

                delta_t = timer() - start_time

                output_value(current_epoch, 'train', '%d/%d' % (step + 1, total_len), '%s/%s' % (gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))), '%.3lf' % (total_loss / (step + 1)), output_info, '\r', config)

            global_step += 1

            # tensorboardX
            # -----
            writer.add_scalar(config.get('output', 'model_name') + '_train_iter', float(loss), global_step)
            # -----

        output_value(current_epoch, 'train', '%d/%d' % (step + 1, total_len), '%s/%s' % (gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))), '%.3lf' % (total_loss / (step + 1)), output_info, None, config)

        if step == -1:
            information = 'There is no data given to the model in this epoch.'
            logger.error(information)
            raise NotImplementedError

        checkpoint(os.path.join(output_path, 'checkpoint_%d.pkl' % current_epoch), model, optimizer, current_epoch, config, global_step)

        # tensorboardX
        # -----
        writer.add_scalar(config.get('output', 'model_name') + '_train_epoch', float(total_loss) / (step + 1), current_epoch)
        # -----

        if current_epoch % test_time == 0:
            with torch.no_grad():
                eval_one(model, valid_dataset, current_epoch, writer, config, gpu_list, output_function, task='valid')

                if do_test:
                    eval_one(model, test_dataset, current_epoch, writer, config, gpu_list, output_function, task='test')


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
        information = 'Cannot save models with error %s.' % str(Exception)
        logger.error(information)
        raise Exception