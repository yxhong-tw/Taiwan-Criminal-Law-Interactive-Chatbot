import json

from legal_judgment_prediction.evaluation import get_micro_macro_prf


def empty_output_function(*args, **kwargs):
    return ''


def bart_output_function(*args, **kwargs):
    total_loss = kwargs['total_loss']
    step = kwargs['step']

    result = {}
    result['average_loss'] = str(total_loss / (step + 1))

    return json.dumps(obj=result, sort_keys=True)


def bert_output_function(*args, **kwargs):
    data = kwargs['data']

    aaa_micro_macro_prf = {}
    aaa_micro_macro_prf['article'] = get_micro_macro_prf(
        data=data['article'])
    aaa_micro_macro_prf['article_source'] = get_micro_macro_prf(
        data=data['article_source'])
    aaa_micro_macro_prf['accusation'] = get_micro_macro_prf(
        data=data['accusation'])
    
    results = {}

    for name in ['article', 'article_source', 'accusation']:
        results[name] = []

        for score_type in ['mip', 'mir', 'mif', 'map', 'mar', 'maf']:
            results[name].append(aaa_micro_macro_prf[name][score_type])

    return json.dumps(obj=results, sort_keys=True)