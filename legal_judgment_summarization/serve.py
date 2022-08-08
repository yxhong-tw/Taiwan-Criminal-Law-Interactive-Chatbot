import logging


logger = logging.getLogger(__name__)


def serve(parameters, *args, **kwargs):
    logger.info('Start to serve.')

    formatter = parameters['formatter']
    model = parameters['model']

    while True:
        text = input('Enter a text: ')
        logger.info(f'The input text: {text}')

        if text == 'shutdown':
            logger.info('Stop to serve.')
            break

        text = formatter(data=text)
        summary = model(data=text, mode='serve')

        print(f'The summary of this text: {summary}')
        logger.info(f'The summary of this text: {summary}')

        print()