import logging

from legal_judgment_prediction.output.functions import \
    null_output_function , basic_output_function \
    , bart_output_function, bert_output_function


logger = logging.getLogger(__name__)


def initialize_output_function(config, *args, **kwargs):
    logger.info('Start to initialize output function.')

    output_function_types = {
        'null': null_output_function,
        'basic': basic_output_function,
        'bart': bart_output_function,
        'bert': bert_output_function
    }

    function_name = config.get('output', 'output_function')

    if function_name in output_function_types:
        logger.info('Initialize output function successfully.')

        return output_function_types[function_name]
    else:
        logger.error(f'There is no function called {function_name}.')
        raise Exception(f'There is no function called {function_name}.')
