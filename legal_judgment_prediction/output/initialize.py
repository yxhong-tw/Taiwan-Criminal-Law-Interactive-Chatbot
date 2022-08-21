import logging

from legal_judgment_prediction.output.functions import \
    empty_output_function, bart_output_function, bert_output_function


logger = logging.getLogger(__name__)


def initialize_output_function(config, *args, **kwargs):
    logger.info('Start to initialize output function.')

    output_function_name = config.get('output', 'output_function')

    output_functions = {
        'empty': empty_output_function
        , 'bart': bart_output_function
        , 'bert': bert_output_function
    }

    if output_function_name in output_functions:
        output_function = output_functions[output_function_name]

        logger.info('Initialize output function successfully.')

        return output_function
    else:
        logger.error(
            f'There is no output function called {output_function_name}.')
        raise Exception(
            f'There is no output function called {output_function_name}.')
