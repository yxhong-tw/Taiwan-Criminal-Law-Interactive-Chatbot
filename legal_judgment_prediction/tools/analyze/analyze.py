import logging

from legal_judgment_prediction.tools.analyze.initialize import initialize_all
from legal_judgment_prediction.tools.analyze.utils import files_analysis, general_analysis, write_back_results


logger = logging.getLogger(__name__)


def analyze(config):
    parameters = initialize_all(config)

    logger.info(f'Start to analyze precedent dataset.')

    results = files_analysis(parameters)
    general_analysis(parameters, results)
    write_back_results(parameters, results)

    logger.info(f'Analyze precedent dataset successfully.')