import json

from legal_judgment_prediction.tools.accuracy import get_micro_macro_prf


def null_output_function(config, data, *args, **kwargs):
    return ''


def basic_output_function(config, data, *args, **kwargs):
    output_value = config.get('output', 'output_value').replace(' ', '').split(',')
    temp = get_micro_macro_prf(data)
    result = {}

    for name in output_value:
        result[name] = temp[name]

    return json.dumps(result, sort_keys=True)


def bart_output_function(config, total_loss, step, *args, **kwargs):
    result = {}
    result['average_loss'] = str(total_loss / (step+1))

    return json.dumps(result, sort_keys=True)


def bert_output_function(config, data, *args, **kwargs):
    temp = {}
    temp['accusation'] = get_micro_macro_prf(data['accusation'])
    temp['article_source'] = get_micro_macro_prf(data['article_source'])
    temp['article'] = get_micro_macro_prf(data['article'])
    
    result = {}
    for name in ['accusation', 'article_source', 'article']:
        result[name] = []

        for score_type in ['mip', 'mir', 'mif', 'map', 'mar', 'maf']:
            result[name].append(temp[name][score_type])

    return json.dumps(result, sort_keys=True)