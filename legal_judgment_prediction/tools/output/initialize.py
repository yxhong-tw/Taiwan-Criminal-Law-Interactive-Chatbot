import logging

from legal_judgment_prediction.tools.output.functions import basic_output_function, null_output_function, bert_output_function


logger = logging.getLogger(__name__)


def initialize_output_function(config, *args, **kwargs):
    output_function_dict = {
        'null': null_output_function,
        'basic': basic_output_function,
        'bert': bert_output_function
    }

    function_name = config.get('output', 'output_function')

    if function_name in output_function_dict:
        return output_function_dict[function_name]
    else:
        # information = 'There is no model called %s.' % function_name
        logger.error(f'There is no function called {function_name}.')
        raise Exception(f'There is no function called {function_name}.')
