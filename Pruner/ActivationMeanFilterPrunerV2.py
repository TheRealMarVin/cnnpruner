from heapq import nsmallest
from operator import itemgetter

import torch

from ModelHelper import get_node_in_model
from Pruner.FilterPruner import FilterPruner

#TODO change the algo so the next comment is working
"""
here we must support the case where we do have an addition of the residual followed by a split and a second residual
block. In this case both branch must be considered for pruning.
"""
class ActivationMeanFilterPrunerV2(FilterPruner):

    def __init__(self, model, sample_run, force_forward_view=False):
        super(ActivationMeanFilterPrunerV2, self).__init__(model, sample_run, force_forward_view)
        self.merged_results = {}
        self.reasign_nodes = {}

    def sort_filters(self, num):
        data = []

        #TODO not the right place but for a test it is a valid attempt
        for k, v in  self.merged_results.items():
            number_to_divide_by = self.connection_count_copy[k]
            divided = torch.div(v, number_to_divide_by)
            for _, v2 in self.reasign_nodes.items():
                for i in v2:
                    self.test_layer_activation[i] = divided

        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((i, j, self.filter_ranks[i][j]))

        res = nsmallest(num, data, itemgetter(2))
        return res

    def post_pruning_plan(self, filters_to_prune_per_layer):
        to_remove = []
        for k, v in self.reasign_nodes.items():
            size = None
            if len(v) == 0:
                #TODO not sure it is supposed to happen... more of a safety for the moment
                to_remove.append(k)
            else:
                for index, elem in enumerate(v):
                    if elem not in filters_to_prune_per_layer:
                        size = 0
                    elif size is None:
                        size = len(filters_to_prune_per_layer[elem])
                    elif len(filters_to_prune_per_layer[elem]) != size:
                        #TODO maybe we should take the biggest or the smallest instead of the previous one
                        filters_to_prune_per_layer[elem] = filters_to_prune_per_layer[v[index - 1]]

        for x in to_remove:
            del self.reasign_nodes[x]
    # def extract_filter_activation_mean(self, out):
    #     for k, v in self.merged_results:
    #         number_to_divide_by = self.connection_count_copy[k]
    #         divided = torch.div(v, number_to_divide_by)
    #         for _, v2 in self.reasign_nodes:
    #             for i in v2:
    #                 self.test_layer_activation[i] = divided
    #
    #     # super().extract_filter_activation_mean(out)

    def handle_after_conv_in_forward(self, curr_module, node_id, out):
        self.conv_layer[node_id] = curr_module
        average_per_batch_item = torch.tensor([[curr.view(-1).mean() for curr in batch_item] for batch_item in out])
        activation_average_sum = torch.sum(average_per_batch_item, dim=0)

        val = activation_average_sum.cuda()
        if node_id not in self.filter_ranks:
            self.test_layer_activation[node_id] = val
        else:
            raise NotImplementedError

        merged_node_id = self.get_id_of_merge(node_id)
        if merged_node_id is not None:
            if merged_node_id not in self.merged_results:
                self.merged_results[merged_node_id] = val
            else:
                self.merged_results[merged_node_id] = self.merged_results[merged_node_id] + val

            #TODO this must be moved in a way that we could handle split with no conv on both side
            print("\t*** Node id: ", node_id) #TODO remove this print
            if merged_node_id not in self.reasign_nodes:
                self.reasign_nodes[merged_node_id] = [node_id]
                print("\t*** ADD Node id: ", node_id)  # TODO remove this print
            else:
                print("\t*** APPEND Node id: ", node_id)  # TODO remove this print
                self.reasign_nodes[merged_node_id].append(node_id)

    def should_ignore_layer(self, layer_id):
        next_id = self.graph[layer_id]
        if next_id not in self.name_dic:
            return True

        #TODO we should do more test like reusing a layer
        return False

    def get_id_of_merge(self, layer_id):
        next_id = self.graph[layer_id]
        if next_id not in self.name_dic:
            return None

        layer = get_node_in_model(self.model, self.name_dic[next_id])

        has_more = True
        if isinstance(layer, torch.nn.modules.conv.Conv2d) or isinstance(layer, torch.nn.modules.Linear):
            has_more = False

        if has_more:
            next_id = self.graph[next_id]
            if next_id not in self.name_dic:
                return None
            elif self.connection_count_copy[next_id] > 1:
                return next_id
            else:
                return self.get_id_of_merge(next_id)

    # def _pre_parse_internal(self, node_name):
    #     curr_module = get_node_in_model(self.model, node_name)
    #     if not isinstance(curr_module, torch.nn.modules.conv.Conv2d):
    #         merged_node_id = self.get_id_of_merge(node_name)
    #         if merged_node_id is not None:
    #             if merged_node_id not in self.reasign_nodes:
    #                 self.reasign_nodes[merged_node_id] = [node_name]
    #                 print("\t*** ADD Node id: ", node_name)  # TODO remove this print
    #             else:
    #                 print("\t*** APPEND Node id: ", node_name)  # TODO remove this print
    #                 self.reasign_nodes[merged_node_id].append(node_name)


