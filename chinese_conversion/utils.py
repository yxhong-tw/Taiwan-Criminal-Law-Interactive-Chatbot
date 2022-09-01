import logging
import os


logger = logging.getLogger(__name__)


def chinese_conversion(parameters):
    logger.info(f'Start to convert chinese with {parameters["config"]} config.')

    for file_name in os.listdir(parameters['source_directory_path']):
        if file_name == 'README.md':
            continue

        logger.info(f'Start to process {file_name}.')

        source_file_path = \
            os.path.join(parameters['source_directory_path'], file_name)

        destination_lines = []

        with open(
                file=source_file_path
                , mode='r'
                , encoding='UTF-8') as file:
            source_lines = file.readlines()

            for line in source_lines:
                line = parameters['converter'].convert(line)
                destination_lines.append(line)

            file.close()

        destination_file_path = \
            os.path.join(parameters['destination_directory_path'], file_name)

        with open(
                file=destination_file_path
                , mode='w'
                , encoding='UTF-8') as file:
            for line in destination_lines:
                file.write(line)

            file.close()

        logger.info(f'Process {file_name} successfully.')

    logger.info(
        f'Convert chinese with {parameters["config"]} config successfully.')