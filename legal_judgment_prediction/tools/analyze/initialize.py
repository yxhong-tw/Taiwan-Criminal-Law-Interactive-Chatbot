import logging


logger = logging.getLogger(__name__)


def initialize_all(config, label):
    results = {}

    results['label'] = label
    results['name'] = config.get(label, 'name')
    results['folder_path'] = config.get(label, 'folder_path')
    results['files_analysis_file_path'] = config.get(label, 'files_analysis_file_path')
    results['general_analysis_file_path'] = config.get(label, 'general_analysis_file_path')
    results['parameters_file_path'] = config.get(label, 'parameters_file_path')

    return results