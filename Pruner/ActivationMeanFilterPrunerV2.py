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
class ActivationMeanFilterPrunerV2(FilterPruner):

    def __init__(self, model, sample_run, force_forward_view=False):
        super(ActivationMeanFilterPrunerV2, self).__init__(model, sample_run, force_forward_view)
        self.merged_results = {}
        # self.reasign_nodes = {}
        self.sets = []
        # self.sets_processed = []

        self.reverse_conv_graph = {}
        self.conv_graph = copy.deepcopy(self.graph)
        self.compute_conv_graph()

    def reset(self):
        super().reset()
        # self.sets_processed = []

    def sort_filters(self, num):
        data = []
        #TODO do we need to reset this one maybe we could save some time
        # self.sets = []

        #TODO not the right place but for a test it is a valid attempt
        # for curr_set in self.sets:
        #     if len(curr_set) <= 1:
        #         continue
        #     set_as_list = list(curr_set)
        #     sum = self.test_layer_activation[set_as_list[0]]
        #     for x in set_as_list[1:]:
        #         sum += self.test_layer_activation[x]
        #
        #     divided = torch.div(sum, len(set_as_list))
        #     for x in set_as_list:
        #         self.test_layer_activation[x] = divided
        # for k, v in self.reverse_conv_graph.items():
        #     if len(v) == 1:
        #         continue
        #
        #     sum = v[0]
        #     for x in v[1:]:
        #         sum += self.test_layer_activation[v]
        #
        #     divided = torch.div(v, number_to_divide_by)
        #
        #
        # for k, v in self.merged_results.items():
        #     number_to_divide_by = self.connection_count_copy[k]
        #     divided = torch.div(v, number_to_divide_by)
        #     for _, v2 in self.reasign_nodes.items():
        #         for i in v2:
        #             self.test_layer_activation[i] = divided

        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((i, j, self.filter_ranks[i][j]))

        res = nsmallest(num, data, itemgetter(2))
        return res

    def extract_filter_activation_mean(self, out):
        for curr_set in self.sets:
            if len(curr_set) <= 1:
                continue
            set_as_list = list(curr_set)
            sum = self.test_layer_activation[set_as_list[0]]
            for x in set_as_list[1:]:
                sum += self.test_layer_activation[x]

            divided = torch.div(sum, len(set_as_list))
            for x in set_as_list:
                self.test_layer_activation[x] = divided

        super().extract_filter_activation_mean(out)

    def post_pruning_plan(self, filters_to_prune_per_layer):
        for curr_set in self.sets:
            if len(curr_set) <= 1:
                continue

            set_as_list = list(curr_set)
            min_val = None
            max_val = 0
            max_index = -1
            for i, elem in enumerate(set_as_list):
                if elem not in filters_to_prune_per_layer:
                    min_val = 0
                    continue

                size = len(filters_to_prune_per_layer[elem])
                if min_val is None or size < min_val:
                    min_val = size
                if size > max_val:
                    max_val = size
                    max_index = i

            if min_val != max_val:
                for i, elem in enumerate(set_as_list):
                    if i == max_index:
                        continue
                    filters_to_prune_per_layer[elem] = filters_to_prune_per_layer[set_as_list[max_index]]

        # TODO make sure every "split" node has the same result has the sibling
        pass
        # to_remove = []
        # for k, v in self.reasign_nodes.items():
        #     size = None
        #     if len(v) == 0:
        #         #TODO not sure it is supposed to happen... more of a safety for the moment
        #         to_remove.append(k)
        #     else:
        #         for index, elem in enumerate(v):
        #             if elem not in filters_to_prune_per_layer:
        #                 size = 0
        #             elif size is None:
        #                 size = len(filters_to_prune_per_layer[elem])
        #             elif len(filters_to_prune_per_layer[elem]) != size:
        #                 #TODO maybe we should take the biggest or the smallest instead of the previous one
        #                 filters_to_prune_per_layer[elem] = filters_to_prune_per_layer[v[index - 1]]
        #
        # for x in to_remove:
        #     del self.reasign_nodes[x]

    """
    The difference with the parent version is that if one of the node in the set was considered. we do not apply
    effect again. It would make the next input smaller than necessary. Let say we remove 3 elements on each branch 
    the input should be only smaller by a size of 3 and not 3 + 3. Although in the case of concat we need to apply 
    effect and reduce the element by the sum of all branches.
    """
    # def prune(self, pruning_dic):
    #     for layer_id, filters_to_remove in pruning_dic.items():
    #         layer = get_node_in_model(self.model, self.name_dic[layer_id])
    #         # print("trying to prune for layer: {} \tID: {}".format(self.name_dic[layer_id], layer_id))
    #         if layer is not None:
    #             initial_filter_count = 0
    #             if isinstance(layer, torch.nn.modules.conv.Conv2d):
    #                 initial_filter_count = self._prune_conv_output_filters(layer, filters_to_remove)
    #
    #             if self._is_node_in_set(layer_id):
    #                 curr_set = self._get_set_for_node(layer_id)
    #                 if curr_set in self.sets_processed:
    #                     continue
    #                 else:
    #                     self.sets_processed.append(curr_set)
    #
    #             if len(filters_to_remove) > 0:
    #                 effect_applied = []
    #                 next_id = self.graph[layer_id]
    #                 for sub_node_id in next_id.split(","):
    #                     if sub_node_id not in effect_applied:
    #                         self._apply_pruning_effect(sub_node_id, filters_to_remove, initial_filter_count,
    #                                                    effect_applied)

    def handle_after_conv_in_forward(self, curr_module, node_id, out):
        self.conv_layer[node_id] = curr_module
        average_per_batch_item = torch.tensor([[curr.view(-1).mean() for curr in batch_item] for batch_item in out])
        activation_average_sum = torch.sum(average_per_batch_item, dim=0)

        val = activation_average_sum.cuda()
        if node_id not in self.filter_ranks:
            self.test_layer_activation[node_id] = val
        else:
            raise NotImplementedError

        # merged_node_id = self.get_id_of_merge(node_id)
        # if node_id is not None:
        #     if node_id not in self.merged_results:
        #         self.merged_results[node_id] = val
        #     else:
        #         self.merged_results[node_id] = self.merged_results[node_id] + val

            # # TODO this must be moved in a way that we could handle split with no conv on both side
            # print("\t*** Node id: ", node_id) #TODO remove this print
            # if node_id not in self.reasign_nodes:
            #     self.reasign_nodes[node_id] = [node_id]
            #     print("\t*** ADD Node id: ", node_id)  # TODO remove this print
            # else:
            #     print("\t*** APPEND Node id: ", node_id)  # TODO remove this print
            #     self.reasign_nodes[node_id].append(node_id)

    def should_ignore_layer(self, layer_id):
        next_id = self.graph[layer_id]
        if next_id not in self.name_dic:
            return True

        # TODO we should do more test like reusing a layer
        return False

    # def get_id_of_merge(self, layer_id):
    #     next_id = self.graph[layer_id]
    #     if next_id not in self.name_dic:
    #         return None
    #
    #     layer = get_node_in_model(self.model, self.name_dic[next_id])
    #
    #     has_more = True
    #     if isinstance(layer, torch.nn.modules.conv.Conv2d) or isinstance(layer, torch.nn.modules.Linear):
    #         has_more = False
    #
    #     if has_more:
    #         next_id = self.graph[next_id]
    #         if next_id not in self.name_dic:
    #             return None
    #         elif self.connection_count_copy[next_id] > 1:
    #             return next_id
    #         else:
    #             return self.get_id_of_merge(next_id)

    def compute_conv_graph(self):
        conv_layers = []
        to_delete = []
        for key, val in self.conv_graph.items():
            module = get_node_in_model(self.model, self.name_dic[key])
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                conv_layers.append(key)
            elif isinstance(module, torch.nn.modules.Linear):
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
        for key, val in self.reverse_conv_graph.items():
            if len(val) == 1:
                continue

            a = 0
            xx = set(val)
            found = False
            for i, j in enumerate(self.sets):
                if len(xx.intersection(j)) > 0:
                    self.sets[i] = xx.union(j)
                    # TODO il y a un probleme avec le la valeur
                    temp[key] = i
                    found = True
                    break

            if not found:
                self.sets.append(xx)
                temp[key] = len(self.sets) - 1

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


