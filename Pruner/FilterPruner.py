import copy
import random
from heapq import nsmallest
from operator import itemgetter

import numpy as np
import torch

from ExecutionGraphHelper import generate_graph, get_input_connection_count_per_entry
from ModelHelper import get_node_in_model


class FilterPruner:
    def __init__(self, model, sample_run):
        self.model = model
        self.activations = {} #TODO remove?
        self.gradients = []
        self.grad_index = 0 # TODO remove
        self.conv_layer = {}
        self.activation_to_layer = {} # TODO remove
        self.filter_ranks = {}
        self.forward_res = {}
        self.activation_index = 0 # TODO remove
        self.test_layer_activation = {} #TODO remove
        self.connection_count = {}
        self.connection_count_copy = {}
        self.features = []
        self.reset()
        model.cpu()
        self.graph, self.name_dic, self.root = generate_graph(model, sample_run)

        model.cuda()


    def reset(self):
        self.activations = {}
        self.gradients = []
        self.features = []
        self.conv_layer = {}
        self.grad_index = 0 # TODO remove
        self.activation_to_layer = {}
        self.filter_ranks = {}
        self.forward_res = {}
        self.activation_index = 0
        self.connection_count = {}
        self.connection_count_copy = {}
        self.test_layer_activation = {} #TODO test

    def parse(self, node_id):
        # print("PARSE node_name: {}".format(node_id))

        node_name = self.name_dic[node_id]
        if self.connection_count[node_id] > 0:
            return None

        curr_module = get_node_in_model(self.model, node_name)
        if curr_module is None:
            # print("is none... should add x together")
            out = self.forward_res[node_id]
        else:
            x = self.forward_res[node_id]
            if isinstance(curr_module, torch.nn.modules.Linear):
                x = x.view(x.size(0), -1)

            out = curr_module(x)
            if isinstance(curr_module, torch.nn.modules.conv.Conv2d):
                self.conv_layer[node_id] = curr_module
                average_per_batch_item = torch.tensor([[curr.view(-1).mean() for curr in batch_item] for batch_item in out])
                activation_average_sum = torch.sum(average_per_batch_item, dim=0)

                val = activation_average_sum.cuda()
                if node_id not in self.filter_ranks:
                    self.test_layer_activation[node_id] = val
                else:
                    self.test_layer_activation[node_id] = self.test_layer_activation[node_id] + val

        res = None
        next_nodes = self.graph[node_id]
        if len(next_nodes) == 0:
            res = out
        else:
            for next_id in self.graph[node_id].split(","):
                self.connection_count[next_id] -= 1
                if next_id in self.forward_res:
                    self.forward_res[next_id] = self.forward_res[next_id] + out
                else:
                    self.forward_res[next_id] = out

                res = self.parse(next_id)
        return res

    # This is super slow because of the way I parse the execution tree, but it works
    def forward(self, x):
        self.activations = {}
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}
        self.forward_res = {}
        self.conv_layer = {}

        self.activation_index = 0

        get_input_connection_count_per_entry(self.graph, self.root, self.connection_count)
        self.connection_count_copy = copy.deepcopy(self.connection_count)
        self.layer_to_parse = self.graph.keys()

        self.connection_count[self.root] = 0    # for the root we have everything we need
        self.forward_res[self.root] = x         # for root we also have the proper input

        x.requires_grad = True
        x = self.parse(self.root)
        return x

    def extract_filter_activation_mean(self, out):
        with torch.no_grad():
            for node_name, curr_module in self.conv_layer.items():
                if self.is_before_merge(node_name):
                    continue


                if node_name not in self.filter_ranks:
                    self.filter_ranks[node_name] = self.test_layer_activation[node_name]
                else:
                    self.filter_ranks[node_name] = self.filter_ranks[node_name] + self.test_layer_activation[node_name]

    def ramdom_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((i, j, self.filter_ranks[i][j]))

        return random.sample(data, num)

    def sort_filters(self, num):
        raise NotImplementedError

    def is_before_merge(self, layer_id):
        next_id = self.graph[layer_id]
        if next_id not in self.name_dic:
            return True

        layer = get_node_in_model(self.model, self.name_dic[next_id])

        has_more = True
        if isinstance(layer, torch.nn.modules.conv.Conv2d) or isinstance(layer, torch.nn.modules.Linear):
            has_more = False

        if has_more:
            next_id = self.graph[next_id]
            if next_id not in self.name_dic:
                return True
            elif self.connection_count_copy[next_id] > 1:
                return True
            else:
                return self.is_before_merge(next_id)

    def normalize_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / torch.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v

    def plan_prunning(self, num_filters_to_prune):
        filters_to_prune = self.sort_filters(num_filters_to_prune)

        filters_to_prune_per_layer = {}
        for (node_id, f, _) in filters_to_prune:
            if node_id not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[node_id] = []
            filters_to_prune_per_layer[node_id].append(f)

        #test if we fully remove a layer
        for node_id in filters_to_prune_per_layer:
            layer_remove_count = len(filters_to_prune_per_layer[node_id])
            in_filter_in_layer = self.conv_layer[node_id].in_channels
            if layer_remove_count == in_filter_in_layer:
                filters_to_prune_per_layer[node_id] = filters_to_prune_per_layer[node_id][:-1]

        #sorting really makes the pruning faster
        for node_id in filters_to_prune_per_layer:
            filters_to_prune_per_layer[node_id] = sorted(filters_to_prune_per_layer[node_id])

        return filters_to_prune_per_layer

    # TODO here we should see what would happen if a layer is fully removed. this is quite annoying
    def prune(self, pruning_dic):
        for layer_id, filters_to_remove in pruning_dic.items():
            layer = get_node_in_model(self.model, self.name_dic[layer_id])
            # print("trying to prune for layer: {} \tID: {}".format(self.name_dic[layer_id], layer_id))
            if layer is not None:
                initial_filter_count = 0
                if isinstance(layer, torch.nn.modules.conv.Conv2d):
                    initial_filter_count = self._prune_conv_output_filters(layer, filters_to_remove)

                if len(filters_to_remove) > 0:
                    effect_applied = []
                    next_id = self.graph[layer_id]
                    for sub_node_id in next_id.split(","):
                        if sub_node_id not in effect_applied:
                            self._apply_pruning_effect(sub_node_id, filters_to_remove, initial_filter_count, effect_applied)

    def _apply_pruning_effect(self, layer_id, removed_filter, initial_filter_count, effect_applied):
        if layer_id not in self.name_dic:
            for sub_node_id in layer_id.split(","):
                self._apply_pruning_effect(sub_node_id, removed_filter, initial_filter_count)
            return
        layer = get_node_in_model(self.model, self.name_dic[layer_id])

        has_more = True
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            self._prune_conv_input_filters(layer, removed_filter, initial_filter_count)
            effect_applied.append(layer_id)
            has_more = False
        elif isinstance(layer, torch.nn.modules.Linear):
            self._prune_input_linear(layer, removed_filter, initial_filter_count)
            effect_applied.append(layer_id)
            has_more = False
        elif isinstance(layer, torch.nn.modules.BatchNorm2d):
            self._prune_conv_input_batchnorm(layer, removed_filter, initial_filter_count)
            effect_applied.append(layer_id)

        if has_more:
            next_id = self.graph[layer_id]
            for sub_node_id in next_id.split(","):
                if sub_node_id not in effect_applied:
                    self._apply_pruning_effect(sub_node_id, removed_filter, initial_filter_count, effect_applied)

    def _prune_conv_output_filters(self, conv, filters_to_remove):
        initial_filter_count = conv.out_channels
        conv.out_channels = conv.out_channels - len(filters_to_remove)
        old_weights = conv.weight.data.detach()
        new_weights = np.delete(old_weights, filters_to_remove, 0)
        conv.weight.data = new_weights
        conv.weight._grad = None

        if conv.bias is not None:
            bias_numpy = conv.bias.data.detach()
            new_bias_numpy = np.delete(bias_numpy, filters_to_remove, 0)
            conv.bias.data = new_bias_numpy
            conv.bias._grad = None

        return initial_filter_count

    def _prune_conv_input_filters(self, conv, removed_filter, _):
        conv.in_channels = conv.in_channels - len(removed_filter)
        old_weights = conv.weight.data
        new_weights = np.delete(old_weights, removed_filter, 1)
        conv.weight.data = new_weights
        # print("conc _ in _ old weight shape {} vs new weight shape {}".format(old_weights.shape, new_weights.shape))
        conv.weight._grad = None

    def _prune_conv_input_batchnorm(self, batchnorm, removed_filter, _):
        batchnorm.num_features = batchnorm.num_features - len(removed_filter)
        old_batch_weights = batchnorm.weight.detach()
        new_batch_weights = np.delete(old_batch_weights, removed_filter, 0)
        batchnorm.weight.data = new_batch_weights

        if batchnorm.bias is not None:
            bias_numpy = batchnorm.bias.data.detach()
            new_bn_bias_numpy = np.delete(bias_numpy, removed_filter, 0)
            batchnorm.bias.data = new_bn_bias_numpy
            batchnorm.bias._grad = None

        batchnorm.weight._grad = None
        if batchnorm.track_running_stats:
            batchnorm.register_buffer('running_mean', torch.zeros(batchnorm.num_features))
            batchnorm.register_buffer('running_var', torch.ones(batchnorm.num_features))
            batchnorm.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            batchnorm.register_parameter('running_mean', None)
            batchnorm.register_parameter('running_var', None)
            batchnorm.register_parameter('num_batches_tracked', None)
        batchnorm.reset_running_stats()
        batchnorm.reset_parameters()

    def _prune_input_linear(self, linear, removed_filter, initial_filter_count):
        lin_in_feat = linear.in_features
        elem_per_channel = (lin_in_feat // initial_filter_count)

        sub_array = [x for x in range(elem_per_channel)]
        weight_to_delete = []
        for filter_index in removed_filter:
            translation = filter_index * elem_per_channel
            weight_to_delete.extend(np.add(translation, sub_array))

        new_lin_in_feat = lin_in_feat - (elem_per_channel * len(removed_filter))
        old_lin_weights = linear.weight.detach()
        lin_new_weigths = np.delete(old_lin_weights, weight_to_delete, 1)
        factor = 1 - (elem_per_channel / lin_in_feat)
        lin_new_weigths.mul_(factor)
        linear.weight.data = lin_new_weigths
        linear.in_features = new_lin_in_feat
        linear.weight._grad = None

    def display_pruning_log(self, pruning_dic):
        layers_pruned = {}
        for layer_index, filter_index in pruning_dic.items():
            layer_name = self.name_dic[layer_index]
            if layer_name not in layers_pruned:
                layers_pruned[layer_name] = 0
                layers_pruned[layer_name] = len(pruning_dic[layer_index])
        print("Layers that will be pruned", layers_pruned)
