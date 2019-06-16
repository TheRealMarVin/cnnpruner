import copy
import random
from heapq import nsmallest
from operator import itemgetter

import numpy as np
import torch
import torch.nn.functional as F

from ExecutionGraphHelper import generate_graph, get_input_connection_count_per_entry
from ModelHelper import get_node_in_model


class FilterPruner:
    def __init__(self, model, sample_run, force_forward_view=False):
        self.model = model
        self.force_forward_view = force_forward_view;
        self.activations = {} #TODO remove?
        self.gradients = []
        self.grad_index = 0 # TODO remove
        self.conv_layer = {}
        self.activation_to_layer = {}
        self.filter_ranks = {}
        self.forward_res = {}
        self.activation_index = 0 # TODO remove
        self.test_layer_activation = {} #TODO rename
        self.connection_count = {}
        self.connection_count_copy = {}
        self.features = []
        self.special_ops_prune_apply_count = {}
        self.reset()
        model.cpu()
        self.graph, self.name_dic, self.root, self.special_op, self.special_op_params = generate_graph(model, sample_run)

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
        self.test_layer_activation = {} #TODO rename
        self.special_ops_prune_apply_count = {}

    def parse(self, node_id):
        node_name = self.name_dic[node_id]
        if self.connection_count[node_id] > 0:
            return None

        curr_module = get_node_in_model(self.model, node_name)
        if curr_module is None:
            if node_id in self.special_op.keys():
                if self.special_op[node_id] == "AveragePool":
                    shape, pad, stride = self.special_op_params[node_id]
                    out = F.avg_pool2d(self.forward_res[node_id], kernel_size=shape, stride=stride)
                elif self.special_op[node_id] == "Add":
                    out = self.forward_res[node_id]
            else:
                out = self.forward_res[node_id]
        else:
            # self._pre_parse_internal(node_id)

            x = self.forward_res[node_id]
            if isinstance(curr_module, torch.nn.modules.Linear):
                x = x.view(x.size(0), -1)

            if isinstance(curr_module, torch.nn.modules.conv.Conv2d):
                self.handle_before_conv_in_forward(curr_module, node_id)

            should_not_skip = True
            if node_id in self.special_op.keys():
                if self.special_op[node_id] == "Concat":
                    should_not_skip = False
                    out = x
                elif self.special_op[node_id] == "AveragePool":
                    shape, pad, stride = self.special_op_params[node_id]
                    out = F.avg_pool2d(x, kernel_size=shape, stride=stride)
                    should_not_skip = False

            if should_not_skip:
                out = curr_module(x)

            if isinstance(curr_module, torch.nn.modules.conv.Conv2d):
                self.handle_after_conv_in_forward(curr_module, node_id, out)

        res = None
        next_nodes = self.graph[node_id]
        if len(next_nodes) == 0:
            res = out
        else:
            for next_id in self.graph[node_id].split(","):
                self.connection_count[next_id] -= 1
                if next_id in self.forward_res:
                    if next_id in self.special_op.keys() and self.special_op[next_id] == "Concat":
                        self.forward_res[next_id] = torch.cat((self.forward_res[next_id], out), 1)
                    else:
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
        if self.force_forward_view:
           x = x.view(x.size(0), x.shape[1])
        return x

    def extract_filter_activation_mean(self, out):
        with torch.no_grad():
            for node_name, curr_module in self.conv_layer.items():
                if self.should_ignore_layer(node_name):
                    continue

                if node_name not in self.filter_ranks:
                    self.filter_ranks[node_name] = self.test_layer_activation[node_name]
                else:
                    self.filter_ranks[node_name] = self.filter_ranks[node_name] + self.test_layer_activation[node_name]

    #TODO fix typo
    def ramdom_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((i, j, self.filter_ranks[i][j]))

        return random.sample(data, num)

    def handle_before_conv_in_forward(self, curr_module, node_id):
        pass

    def handle_after_conv_in_forward(self, curr_module, node_id, out):
        pass

    def post_run_cleanup(self):
        pass

    #TODO might not need this one
    def post_pruning_plan(self, filters_to_prune_per_layer):
        pass

    # def _pre_parse_internal(self, node_id):
    #     pass

    def sort_filters(self, num):
        raise NotImplementedError

    """
    This must be called once the first forward pass is done
    """
    def get_number_of_filter_to_prune(self):
        total_filter_count = 0
        for k, conv in self.conv_layer.items():
            if self.should_ignore_layer(k):
                continue

            total_filter_count = total_filter_count + conv.out_channels

        return total_filter_count

    #TODO we should change this method to only test the most restrictive case
    def should_ignore_layer(self, layer_id):
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
                return self.should_ignore_layer(next_id)

    def normalize_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / torch.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v

    def plan_pruning(self, num_filters_to_prune):
        filters_to_prune = self.sort_filters(num_filters_to_prune)

        filters_to_prune_per_layer = {}
        for (node_id, f, _) in filters_to_prune:
            if node_id not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[node_id] = []
            filters_to_prune_per_layer[node_id].append(f)

        # TODO test if we fully remove a layer
        for node_id in filters_to_prune_per_layer:
            layer_remove_count = len(filters_to_prune_per_layer[node_id])
            in_filter_in_layer = self.conv_layer[node_id].in_channels
            if layer_remove_count == in_filter_in_layer:
                filters_to_prune_per_layer[node_id] = filters_to_prune_per_layer[node_id][:-1]

        # sorting really makes the pruning faster
        for node_id in filters_to_prune_per_layer:
            filters_to_prune_per_layer[node_id] = sorted(filters_to_prune_per_layer[node_id])

        self.post_pruning_plan(filters_to_prune_per_layer)
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

    """
        we apply propagation. One thing that is tricky is that some feature like add should not propagate 
        more than once. If we take add as an example only one path should propagate effect past his node
        otherwise the input of the next node will not be right. However some node like concat shoud apply
        reduction twice and in that case we have to do some magic to offset. Beware of this magic trick!
        
    """
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
            initial_filter_count = layer.num_features
        else:
            if layer_id in self.special_op:
                if layer_id not in self.special_ops_prune_apply_count.keys():
                    self.special_ops_prune_apply_count[layer_id] = 0
                else:

                    size = self.special_ops_prune_apply_count[layer_id]
                    self.special_ops_prune_apply_count[layer_id] = size + 1
                    if self.special_op[layer_id] == "Add":
                        has_more = False
                    # TODO handle concat properly
                    # elif self.special_op[layer_id] == "Concat":
                    #     for

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
        after_pruning = {}
        sum = 0
        remain_sum = 0
        for layer_index, filter_index in pruning_dic.items():
            layer_name = self.name_dic[layer_index]
            if layer_name not in layers_pruned:
                to_remove_count = len(pruning_dic[layer_index])
                sum = sum + to_remove_count
                remain_sum = remain_sum + self.conv_layer[layer_index].out_channels - to_remove_count
                layers_pruned[layer_name] = to_remove_count
                after_pruning[layer_name] = self.conv_layer[layer_index].out_channels - to_remove_count
        print("Layers that will be pruned", layers_pruned)
        print("convolution remaining after pruning", after_pruning)
