import logging
import torch.nn as nn


logger = logging.getLogger(__name__)


class MultiLabelSoftmaxLoss(nn.Module):
    def __init__(self, task_number=0):
        super(MultiLabelSoftmaxLoss, self).__init__()
        
        self.criterion = []

        for _ in range(task_number):
            self.criterion.append(nn.CrossEntropyLoss())


    # The size of outputs is [batch_size, task_number, 2].
    def forward(self, outputs, labels):
        loss = 0

        # The value of outputs.size()[1] is task_number.
        for task_index in range(outputs.size()[1]):
            # The size of outputs[:, task_index, :] is [batch_size, 2]
            # The size of one_task_outputs is [batch_size, 2]
            # one_task_outputs = \
            #     outputs[:, task_index, :].view(outputs.size()[0], -1)
            one_task_outputs = outputs[:, task_index, :]

            # The size of labels[:, task_index] is [batch_size]
            one_task_labels = labels[:, task_index]

            loss += self.criterion[task_index](
                one_task_outputs
                , one_task_labels)

        return loss