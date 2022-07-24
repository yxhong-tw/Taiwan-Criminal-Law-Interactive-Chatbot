import logging

from legal_judgment_prediction.tools.generate.initialize import initialize_all
from legal_judgment_prediction.tools.generate.utils import top_50_article, all_article


logger = logging.getLogger(__name__)


# label -> data_type, range -> mode
def generate(config, label, range):
    parameters = initialize_all(config, label, range)

    if range == 'top_50_article':
        top_50_article(parameters)
    elif range == 'all_article':
        all_article(parameters)
    else:
        raise Exception(f'There is no range named {range}.')