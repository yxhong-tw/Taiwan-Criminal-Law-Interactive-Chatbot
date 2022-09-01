import logging
import torch


logger = logging.Logger(__name__)


def get_prf(data, *args, **kwargs):
    if data['TP'] == 0:
        if data['FP'] == 0 and data['FN'] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = (1.0 * data['TP'] / (data['TP'] + data['FP']))
        recall = (1.0 * data['TP'] / (data['TP'] + data['FN']))
        f1 = (2 * precision * recall / (precision + recall))

    return precision, recall, f1


def get_micro_macro_prf(data, *args, **kwargs):
    precision = []
    recall = []
    f1 = []

    total = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

    for one_data in range(len(data)):
        total['TP'] += data[one_data]['TP']
        total['FP'] += data[one_data]['FP']
        total['FN'] += data[one_data]['FN']
        total['TN'] += data[one_data]['TN']

        p, r, f = get_prf(data=data[one_data])

        precision.append(p)
        recall.append(r)
        f1.append(f)

    micro_precision, micro_recall, micro_f1 = get_prf(data=total)

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0

    for index in range(len(f1)):
        macro_precision += precision[index]
        macro_recall += recall[index]
        macro_f1 += f1[index]

    macro_precision /= len(f1)
    macro_recall /= len(f1)
    macro_f1 /= len(f1)

    return {
        'mip': round(micro_precision, 3)
        , 'mir': round(micro_recall, 3)
        , 'mif': round(micro_f1, 3)
        , 'map': round(macro_precision, 3)
        , 'mar': round(macro_recall, 3)
        , 'maf': round(macro_f1, 3)
    }


# TODO
# need review and understand what this part do
def multi_label_accuracy(outputs, label, result=None, *args, **kwargs):
    if len(label[0]) != len(outputs[0]):
    # if outputs.size() != label.size()
        logger.error('The dimension of predictions and labels does not match.')
        raise Exception(
            'The dimension of predictions and labels does not match.')

    if len(outputs.size()) > 2:
        outputs = outputs.view(outputs.size()[0], -1, 2)
        outputs = torch.nn.Softmax(dim=2)(outputs)
        outputs = outputs[:, :, 1]

    # outputs = outputs.view(outputs.size()[0], -1, 2)
    # outputs = torch.nn.Softmax(dim=2)(outputs)
    # outputs = outputs[:, :, 1]

    # According to https://blog.csdn.net/DreamHome_S/article/details/85259533
    # , using .detach() to instead of .data
    outputs = outputs.data
    labels = label.data
    # outputs = outputs.detach()
    # labels = label.detach()
    # labels = label

    if result is None:
        result = []

    total = 0
    nr_classes = outputs.size(1)

    while len(result) < nr_classes:
        result.append({'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0})

    for i in range(nr_classes):
        outputs1 = (outputs[:, i] >= 0.5).long()
        labels1 = (labels[:, i].float() >= 0.5).long()
        total += int((labels1 * outputs1).sum())
        total += int(((1 - labels1) * (1 - outputs1)).sum())

        if result is None:
            continue

        result[i]['TP'] += int((labels1 * outputs1).sum())
        result[i]['FN'] += int((labels1 * (1 - outputs1)).sum())
        result[i]['FP'] += int(((1 - labels1) * outputs1).sum())
        result[i]['TN'] += int(((1 - labels1) * (1 - outputs1)).sum())

    return result
