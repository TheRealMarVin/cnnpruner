import random

import torch

from Pruner.FilterPruner import FilterPruner


class RandomFilterPruner(FilterPruner):

    def __init__(self, model, sample_run):
        super(RandomFilterPruner, self).__init__(model, sample_run)

    def sort_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((i, j, self.filter_ranks[i][j]))

        return random.sample(data, num)

    def handle_after_conv_in_forward(self, curr_module, node_id, out):
        #we just reuse mean activation
        self.conv_layer[node_id] = curr_module
        average_per_batch_item = torch.tensor([[curr.view(-1).mean() for curr in batch_item] for batch_item in out])
        activation_average_sum = torch.sum(average_per_batch_item, dim=0)

        val = activation_average_sum.cuda()
        if node_id not in self.filter_ranks:
            self.test_layer_activation[node_id] = val
        else:
            # TODO probably not necessary
            self.test_layer_activation[node_id] = self.test_layer_activation[node_id] + val
