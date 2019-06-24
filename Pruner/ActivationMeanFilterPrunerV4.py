import copy
import sys
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
class ActivationMeanFilterPrunerV4(FilterPruner):

    def __init__(self, model, sample_run, force_forward_view=False):
        super(ActivationMeanFilterPrunerV4, self).__init__(model, sample_run, force_forward_view)
        self.merged_results = {}
        # self.reasign_nodes = {}
        self.sets = []
        self.ignore_list = []
        # self.sets_processed = []

        self.reverse_conv_graph = {}
        self.conv_graph = copy.deepcopy(self.graph_res.execution_graph)
        self.compute_conv_graph()

    def reset(self):
        super().reset()
        # self.sets_processed = []

    def sort_filters(self, num):
        data = []

        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((i, j, self.filter_ranks[i][j]))

        res = nsmallest(num, data, itemgetter(2))
        return res

    # def extract_filter_activation_mean(self, out):
    #     for curr_set in self.sets:
    #         if len(curr_set) <= 1:
    #             continue
    #         set_as_list = list(curr_set)
    #         sum = self.test_layer_activation[set_as_list[0]]
    #         for x in set_as_list[1:]:
    #             sum += self.test_layer_activation[x]
    #
    #         divided = sum # TODO just a test
    #         # divided = torch.div(sum, len(set_as_list))
    #         for x in set_as_list:
    #             self.test_layer_activation[x] = divided
    #
    #     super().extract_filter_activation_mean(out)

    def post_pruning_plan(self, filters_to_prune_per_layer):
        for curr_set in self.sets:
            if len(curr_set) <= 1:
                continue

            set_as_list = list(curr_set)
            intersect_set = None
            for i, elem in enumerate(set_as_list):
                if i == 0:
                    intersect_set = set(filters_to_prune_per_layer[elem])
                else:
                    intersect_set = intersect_set.intersection(set(filters_to_prune_per_layer[elem]))

            if intersect_set is None:
                for elem in set_as_list:
                    del filters_to_prune_per_layer[elem]
            else:
                for elem in set_as_list:
                    filters_to_prune_per_layer[elem] = list(intersect_set)

    def handle_after_conv_in_forward(self, curr_module, node_id, out):
        self.conv_layer[node_id] = curr_module
        average_per_batch_item = torch.tensor([[curr.view(-1).mean() for curr in batch_item] for batch_item in out])
        activation_average_sum = torch.sum(average_per_batch_item, dim=0)

        val = activation_average_sum.cuda()
        if node_id not in self.filter_ranks:
            self.test_layer_activation[node_id] = val
        else:
            raise NotImplementedError

    def should_ignore_layer(self, layer_id):
        next_id = self.graph_res.execution_graph[layer_id]
        if next_id not in self.graph_res.name_dict:
            return True

        if layer_id in self.ignore_list:
            return True

        # TODO we should do more test like reusing a layer
        return False

    def compute_conv_graph(self):
        conv_layers = []
        to_delete = []
        for key, val in self.conv_graph.items():
            module = get_node_in_model(self.model, self.graph_res.name_dict[key])
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                conv_layers.append(key)
            elif key == self.graph_res.out_node:
                conv_layers.append(key)
            else:
                to_delete.append(key)

        for key in conv_layers:
            next = self.conv_graph[key]
            self.conv_graph[key] = ""
            new_next = []
            if next != "":
                for elem in next.split(","):
                    res = self._get_next_conv_id(conv_layers, elem)
                    new_next.extend(res)

            self.conv_graph[key] = ",".join(new_next)

        for key in to_delete:
            self.conv_graph.pop(key, None)

        self.reverse_conv_graph = {}
        for key, val in self.conv_graph.items():
            for x in val.split(","):
                if x not in self.reverse_conv_graph:
                    self.reverse_conv_graph[x] = [key]
                else:
                    self.reverse_conv_graph[x].append(key)

        #TODO ca c'est ce qu'on va vouloir garder dans notre classe
        temp = {}
        self.sets = []
        elem_to_del = None
        for key, val in self.reverse_conv_graph.items():
            if len(val) == 1:
                continue

            array_as_set = set(val)
            found = False
            for i, j in enumerate(self.sets):
                if len(array_as_set.intersection(j)) > 0:
                    self.sets[i] = array_as_set.union(j)
                    if key == self.graph_res.out_node:
                        elem_to_del = i
                    temp[key] = i
                    found = True
                    break

            if not found:
                self.sets.append(array_as_set)
                temp[key] = len(self.sets) - 1
                if key == self.graph_res.out_node:
                    elem_to_del = len(self.sets) - 1

        if elem_to_del is not None:
            self.ignore_list = list(self.sets[elem_to_del])
            del self.sets[elem_to_del]

        # print("sets:", self.sets)
        # print("temp:", temp)
        # to_remove = []
        # for i, curr_set in enumerate(self.sets):
        #     if len(curr_set.intersection(self.graph_res.out_node)) > 0:
        #         to_remove.append(i)
        #
        # for elem_to_remove in to_remove:
        #     del self.sets[elem_to_remove]

        a = 0

    def _get_next_conv_id(self, conv_layers, node):
        res = []
        next = self.conv_graph[node]
        if next != "":
            for elem in next.split(","):
                if elem in conv_layers:
                    res.append(elem)
                else:
                    res.extend(self._get_next_conv_id(conv_layers, elem))

        return res

    def _get_set_for_node(self, node_id):
        for x in self.sets:
            if node_id in x:
                return x

        return None

    def _is_node_in_set(self, node_id):
        for x in self.sets:
            if node_id in x:
                return True

        return False


