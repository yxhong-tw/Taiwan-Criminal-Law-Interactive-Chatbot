import json


def bart_output_function(total_loss, step, *args, **kwargs):
    result = {}
    result['average_loss'] = str(total_loss / (step+1))

    return json.dumps(result, sort_keys=True)