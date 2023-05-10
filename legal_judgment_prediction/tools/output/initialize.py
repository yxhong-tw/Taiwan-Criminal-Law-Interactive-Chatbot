import logging

from legal_judgment_prediction.tools.output.functions import basic_output_function, null_output_function, ljp_output_function


logger = logging.getLogger(__name__)


def init_output_function(config, *args, **params):
    output_function_dict = {
        'Null': null_output_function,
        'Basic': basic_output_function,
        'LJP': ljp_output_function
    }

    function_name = config.get('output', 'output_function')

    if function_name in output_function_dict:
        return output_function_dict[function_name]
    else:
        information = 'There is no model called %s.' % function_name
        logger.error(information)
        raise AttributeError