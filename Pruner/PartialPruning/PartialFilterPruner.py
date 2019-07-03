import torch

from ModelHelper import get_node_in_model
from Pruner.FilterPruner import FilterPruner


class PartialFilterPruner(FilterPruner):
    def __init__(self, model, sample_run, force_forward_view=False,
                 ignore_last_conv=False):
        super(PartialFilterPruner, self).__init__(model, sample_run, force_forward_view, ignore_last_conv)

    def should_ignore_layer(self, layer_id):
        next_id = self.graph_res.execution_graph[layer_id]
        if next_id not in self.graph_res.name_dict:
            return True

        layer = get_node_in_model(self.model, self.graph_res.name_dict[next_id])

        has_more = True
        if isinstance(layer, torch.nn.modules.conv.Conv2d) or isinstance(layer, torch.nn.modules.Linear):
            has_more = False

        if has_more:
            next_id = self.graph_res.execution_graph[next_id]
            if next_id not in self.graph_res.name_dict:
                return True
            elif self.connection_count_copy[next_id] > 1:
                return True
            else:
                return self.should_ignore_layer(next_id)
