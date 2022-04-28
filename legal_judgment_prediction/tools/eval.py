import logging
import torch

from timeit import default_timer as timer
from torch.autograd import Variable

from legal_judgment_prediction.tools.utils import gen_time_str, output_value


logger = logging.getLogger(__name__)


def eval(parameters, config, gpu_list):
    model = parameters['model']
    output_function = parameters['output_function']
    test_dataset = parameters['test_dataset']

    eval_one(model, test_dataset, 0, None, config, gpu_list, output_function, task='test')


def eval_one(model, dataset, epoch, writer, config, gpu_list, output_function, task):
    model.eval()

    acc_result = None
    total_loss = 0
    cnt = 0
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

        results = model(data, config, gpu_list, acc_result, 'valid')

        loss, acc_result = results['loss'], results['acc_result']
        total_loss += float(loss)
        cnt += 1

        if step % output_time == 0:
            delta_t = timer() - start_time

            output_value(epoch, task, '%d/%d' % (step + 1, total_len), '%s/%s' % (gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))), '%.3lf' % (total_loss / (step + 1)), output_info, '\r', config)


    if step == -1:
        information = 'There is no data given to the model in this epoch, check your data.'
        logger.error(information)
        raise NotImplementedError

    delta_t = timer() - start_time
    output_info = output_function(acc_result, config)

    output_value(epoch, task, '%d/%d' % (step + 1, total_len), '%s/%s' % (gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))), '%.3lf' % (total_loss / (step + 1)), output_info, None, config)


    if task == 'valid':
        writer.add_scalar(config.get('output', 'model_name') + '_eval_epoch', float(total_loss) / (step + 1), epoch)
        model.train()
