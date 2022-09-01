import logging

from legal_judgment_summarization.output.functions import bart_output_function


logger = logging.getLogger(__name__)


def initialize_output_function(config, *args, **kwargs):
    logger.info('Start to initialize output function.')

    output_function_types = {'bart': bart_output_function}

    function_name = config.get('output', 'output_function')

    if function_name in output_function_types:
        logger.info('Initialize output function successfully.')

        return output_function_types[function_name]
    else:
        logger.error(f'There is no function called {function_name}.')
        raise Exception(f'There is no function called {function_name}.')
