from heapq import nsmallest
from operator import itemgetter

import torch

from Pruner.PartialPruning.PartialFilterPruner import PartialFilterPruner


class TaylorExpansionFilterPruner(PartialFilterPruner):

    def __init__(self, model, sample_run, force_forward_view=False, ignore_last_conv=False):
        super(TaylorExpansionFilterPruner, self).__init__(model, sample_run, force_forward_view, ignore_last_conv)
        self.handles = {}

    def sort_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((i, j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def handle_before_conv_in_forward(self, curr_module, node_id):
        handle = curr_module.register_backward_hook(self.estimate_taylor)
        self.handles[node_id] = handle
        self.activation_to_layer[node_id] = curr_module

    def handle_after_conv_in_forward(self, curr_module, node_id, out):
        self.conv_layer[node_id] = curr_module
        self.activations[node_id] = out

    def post_run_cleanup(self):
        for _, handle in self.handles.items():
            handle.remove()

        self.handles = {}

    def estimate_taylor(self, module, grad_input, grad_output):
        node_id = -1
        for k, v in self.activation_to_layer.items():
            if v == module:
                node_id = k
                break

        if node_id == -1:
            return
        batch_size, _, filter_width, filter_height = self.activations[node_id].size()
        param_count = batch_size * filter_width * filter_height

        # we must ignore dim 1 because it is the input size
        estimates = self.activations[node_id].mul_(grad_output[0]).sum(dim=3).sum(dim=2).sum(dim=0).div_(param_count)

        self.test_layer_activation[node_id] = torch.abs(estimates) / torch.sqrt(torch.sum(estimates * estimates))

