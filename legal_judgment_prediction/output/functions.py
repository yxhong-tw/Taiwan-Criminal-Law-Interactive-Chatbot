import json

from legal_judgment_prediction.evaluation import get_micro_macro_prf


def null_output_function(config, data, *args, **kwargs):
    return ''


def basic_output_function(config, data, *args, **kwargs):
    output_value = \
        config.get('output', 'output_value').replace(' ', '').split(',')
    temp = get_micro_macro_prf(data)

    results = {}

    for name in output_value:
        results[name] = temp[name]

    return json.dumps(results, sort_keys=True)


def bart_output_function(config, total_loss, step, *args, **kwargs):
    result = {}
    result['average_loss'] = str(total_loss / (step + 1))

    return json.dumps(result, sort_keys=True)


def bert_output_function(config, data, *args, **kwargs):
    temp = {}
    temp['article'] = get_micro_macro_prf(data['article'])
    temp['article_source'] = get_micro_macro_prf(data['article_source'])
    temp['accusation'] = get_micro_macro_prf(data['accusation'])
    
    results = {}

    for name in ['article', 'article_source', 'accusation']:
        results[name] = []

        for score_type in ['mip', 'mir', 'mif', 'map', 'mar', 'maf']:
            results[name].append(temp[name][score_type])

    return json.dumps(results, sort_keys=True)