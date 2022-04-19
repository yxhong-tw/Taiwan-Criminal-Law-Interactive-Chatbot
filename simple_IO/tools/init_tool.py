import torch

from simple_IO.reader.reader import init_formatter
from simple_IO.model import get_model


def init_all(config, gpu_list, checkpoint, *args, **params):
    result = {}

    init_formatter(config, ['serve'], *args, **params)

    model = get_model(config.get('model', 'model_name'))(config, gpu_list, *args, **params)

    if len(gpu_list) > 0:
        model = model.cuda()

        try:
            model.init_multi_gpu(gpu_list, config, *args, **params)
        except Exception as e:
            information = "No init_multi_gpu implemented in the model, use single gpu instead."
            print(information)

    try:
        parameters = torch.load(checkpoint)
        model.load_state_dict(parameters['model'])
    except Exception as e:
        information = "Cannot load checkpoint file with error %s" % str(e)
        print(information)
        raise e

    result["model"] = model

    return result
