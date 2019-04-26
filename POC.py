import copy
import os
import random
from heapq import nsmallest
from operator import itemgetter

import numpy as np
import torch

from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.models.resnet import BasicBlock
from torchvision.transforms import transforms

from deeplib_ext.CustomDeepLib import train, test
from ExecutionGraphHelper import generate_graph, get_input_connection_count_per_entry
from FileHelper import load_obj, save_obj
from ModelHelper import get_node_in_model, total_num_filters
from deeplib_ext.MultiHistory import MultiHistory
from deeplib_ext.history import History
from models.AlexNetSki import alexnetski

# TODO check this one!!! https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
# and this: https://github.com/fg91/visualizing-cnn-feature-maps/blob/master/Calculate_mean_activation_per_filter_in_specific_layer_given_an_image.ipynb
from models.FResiNet import FResiNet


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

    # #TODO most probably useless
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
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((i, j, self.filter_ranks[i][j]))

        # return random.sample(data, num)
        return nsmallest(num, data, itemgetter(2))
        # return nlargest(num, data, itemgetter(2))

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
###

def common_training_code(model,
                         pruned_save_path=None,
                         best_result_save_path=None,
                         pruned_best_result_save_path=None,
                         retrain_if_weight_loaded=False,
                         sample_run=None,
                         reuse_cut_filter=False,
                         max_percent_per_iteration=0.1,
                         prune_ratio=0.3,
                         n_epoch=10,
                         n_epoch_retrain=3,
                         n_epoch_total=20):
    test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = CIFAR10("C:/dev/data/cifar10/", train=True, transform=test_transform, download=True)
    test_dataset = CIFAR10("C:/dev/data/cifar10/", train=False, transform=test_transform, download=True)

    # display_sample_data(train_dataset)
    # display_sample_data(test_dataset)
    model.cuda()

    use_gpu = True
    batch_size = 32
    learning_rate = 0.01

    history = History()

    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.007)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = None
    # scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

    #
    should_train = True
    if best_result_save_path is not None:
        if os.path.isfile(best_result_save_path):
            model.load_state_dict(torch.load(best_result_save_path))
            if not retrain_if_weight_loaded:
                should_train = False
    if should_train:
        local_history = train(model, optimizer, train_dataset, n_epoch,
                              batch_size, use_gpu=use_gpu, criterion=criterion,
                              scheduler=scheduler, best_result_save_path=best_result_save_path)
        history.append(local_history)
        n_epoch_total = n_epoch_total - n_epoch
        # history.display()

    test_score = test(model, test_dataset, batch_size, use_gpu=use_gpu)
    print('Test:\n\tScore: {}'.format(test_score))

    ###
    #TODO maybe put the loop content in a function that looks terrible now
    if prune_ratio is not None:
        pruner = FilterPruner(model, sample_run)
        number_of_filters = total_num_filters(model)
        filter_to_prune = int(number_of_filters * prune_ratio)
        max_filters_to_prune_on_iteration = int(number_of_filters * max_percent_per_iteration)
        if filter_to_prune < max_filters_to_prune_on_iteration:
            max_filters_to_prune_on_iteration = filter_to_prune
        iterations = (filter_to_prune//max_filters_to_prune_on_iteration)
        print("{} iterations to reduce {:2.2f}% filters".format(iterations, prune_ratio*100))

        for param in model.parameters():
            param.requires_grad = True

        for iteration_idx in range(iterations):
            print("Perform pruning iteration: {}".format(iteration_idx))
            pruner.reset()

            prune_targets = None
            if reuse_cut_filter:
                prune_targets = load_obj("filters_dic")

            if prune_targets is None:
                train(model, optimizer, train_dataset, 1, batch_size, use_gpu=use_gpu, criterion=criterion,
                      scheduler=scheduler, pruner=pruner, batch_count=1, should_validate=False)

                pruner.normalize_layer()

                prune_targets = pruner.plan_prunning(max_filters_to_prune_on_iteration)
                if reuse_cut_filter:
                    save_obj(prune_targets, "filters_dic")

            pruner.display_pruning_log(prune_targets)

            print("Pruning filters.. ")
            model = model.cpu()
            pruner.prune(prune_targets)

            model = model.cuda()
            # optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.007)
            optimizer = torch.optim.SGD(model.parameters(), learning_rate)
            pruner.reset()

            print("Filters pruned {}%".format(100 - (100 * float(total_num_filters(model)) / number_of_filters)))
            new_test_score = test(model, test_dataset, batch_size, use_gpu=use_gpu)
            print('Test:\n\tpost prune Score: {}'.format(new_test_score))

            basedir = os.path.dirname(pruned_save_path)
            if not os.path.exists(basedir):
                os.makedirs(basedir)
            torch.save(model, pruned_save_path)

            print("Fine tuning to recover from prunning iteration.")
            local_history = train(model, optimizer, train_dataset, n_epoch_retrain, batch_size, use_gpu=use_gpu,
                                  criterion=None, scheduler=scheduler, pruner=None,
                                  best_result_save_path=pruned_best_result_save_path)
            n_epoch_total = n_epoch_total - n_epoch_retrain
            history.append(local_history)
            # local_history.display()
            test_score = test(model, test_dataset, batch_size, use_gpu=use_gpu)
            print('Test pruning iteration :{}\n\tScore: {}'.format(iteration_idx, test_score))

    ###
    history.display()

    test_score = test(model, test_dataset, batch_size, use_gpu=use_gpu)
    print('Test Fin :\n\tScore: {}'.format(test_score))

    if n_epoch_total > 0:
        local_history = train(model, optimizer, train_dataset, n_epoch_total,
                              batch_size, use_gpu=use_gpu, criterion=criterion,
                              scheduler=scheduler, best_result_save_path=pruned_best_result_save_path)
        history.append(local_history)

    return history


def exec_alexnet(max_percent_per_iteration=0.1, prune_ratio=0.3, n_epoch=10):
    print("***alexnet")
    model = alexnetski(pretrained=True)
    model.cuda()

    history = common_training_code(model, pruned_save_path="saved/alex{}/PrunedAlexnet.pth".format(prune_ratio),
                                   # best_result_save_path="saved/alex{}/alexnet.pth".format(prune_ratio),
                                   pruned_best_result_save_path="saved/alex{}/alexnet_pruned.pth".format(prune_ratio),
                                   sample_run=torch.zeros([1, 3, 224, 224]),
                                   reuse_cut_filter=False,
                                   max_percent_per_iteration=max_percent_per_iteration,
                                   prune_ratio=prune_ratio,
                                   n_epoch=n_epoch,
                                   n_epoch_retrain=2,
                                   n_epoch_total=20)

    return history


def exec_dense_net(max_percent_per_iteration=0.1, prune_ratio=0.3, n_epoch=10):
    print("***densenet121")

    model = models.densenet121(num_classes=10)

    model.cuda()

    history = common_training_code(model, pruned_save_path="saved/resnet18/Prunedresnet.pth,",
                                   # best_result_save_path="saved/resnet18/resnet18.pth",
                                   pruned_best_result_save_path="saved/resnet18/resnet18_pruned.pth",
                                   sample_run=torch.zeros([1, 3, 224, 224]),
                                   reuse_cut_filter=False,
                                   max_percent_per_iteration=max_percent_per_iteration,
                                   prune_ratio=prune_ratio,
                                   n_epoch=n_epoch,
                                   n_epoch_retrain=2,
                                   n_epoch_total=20)
    return history


def exec_vgg16(max_percent_per_iteration=0.1, prune_ratio=0.3, n_epoch=10):
    print("***vgg16")

    model = models.vgg16(num_classes=10)

    model.cuda()

    history = common_training_code(model, pruned_save_path="saved/resnet18/Prunedresnet.pth,",
                                   # best_result_save_path="saved/resnet18/resnet18.pth",
                                   pruned_best_result_save_path="saved/resnet18/resnet18_pruned.pth",
                                   sample_run=torch.zeros([1, 3, 224, 224]),
                                   reuse_cut_filter=False,
                                   max_percent_per_iteration=max_percent_per_iteration,
                                   prune_ratio=prune_ratio,
                                   n_epoch=n_epoch,
                                   n_epoch_retrain=2,
                                   n_epoch_total=20)
    return history


def exec_resnet18(max_percent_per_iteration=0.1, prune_ratio=0.3, n_epoch=10):
    print("***resnet 18")

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    model.cuda()

    history = common_training_code(model, pruned_save_path="saved/resnet18/Prunedresnet.pth,",
                                   # best_result_save_path="saved/resnet18/resnet18.pth",
                                   pruned_best_result_save_path="saved/resnet18/resnet18_pruned.pth",
                                   sample_run=torch.zeros([1, 3, 224, 224]),
                                   reuse_cut_filter=False,
                                   max_percent_per_iteration=max_percent_per_iteration,
                                   prune_ratio=prune_ratio,
                                   n_epoch=n_epoch,
                                   n_epoch_retrain=2,
                                   n_epoch_total=20)
    return history


def exec_resnet34(max_percent_per_iteration=0.1, prune_ratio=0.3, n_epoch=10):
    print("***resnet 34")

    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    model.cuda()

    history = common_training_code(model, pruned_save_path="saved/resnet34/Prunedresnet.pth,",
                                   # best_result_save_path="saved/resnet34/resnet18.pth",
                                   pruned_best_result_save_path="saved/resnet34/resnet34_pruned.pth",
                                   sample_run=torch.zeros([1, 3, 224, 224]),
                                   reuse_cut_filter=False,
                                   max_percent_per_iteration=max_percent_per_iteration,
                                   prune_ratio=prune_ratio,
                                   n_epoch=n_epoch,
                                   n_epoch_retrain=2,
                                   n_epoch_total=20)
    return history


def exec_resnet50(max_percent_per_iteration=0.1, prune_ratio=0.3, n_epoch=10):
    print("***resnet 50")

    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    model.cuda()

    history = common_training_code(model, pruned_save_path="saved/resnet50/Prunedresnet.pth,",
                                   # best_result_save_path="saved/resnet50/resnet18.pth",
                                   pruned_best_result_save_path="saved/resnet50/resnet50_pruned.pth",
                                   sample_run=torch.zeros([1, 3, 224, 224]),
                                   reuse_cut_filter=False,
                                   max_percent_per_iteration=max_percent_per_iteration,
                                   prune_ratio=prune_ratio,
                                   n_epoch=n_epoch,
                                   n_epoch_retrain=2,
                                   n_epoch_total=20)
    return history


def run_startegy_prune_compare():
    multi_history = MultiHistory()
    h = exec_resnet18(max_percent_per_iteration=0.0, prune_ratio=None, n_epoch=10)
    multi_history.append_history("Resnet 18-0", h)
    h = exec_resnet18(max_percent_per_iteration=0.2, prune_ratio=0.2, n_epoch=10)
    multi_history.append_history("Resnet 18-20", h)
    # h = exec_resnet34()
    # multi_history.append_history("Resnet 34", h)
    # h = exec_resnet50()
    # multi_history.append_history("Resnet 50", h)
    h = exec_alexnet(max_percent_per_iteration=0.0, prune_ratio=None, n_epoch=20)
    multi_history.append_history("Alexnet 0", h)
    h = exec_alexnet(max_percent_per_iteration=0.2, prune_ratio=0.2, n_epoch=20)
    multi_history.append_history("Alexnet 20", h)

    h = exec_vgg16(max_percent_per_iteration=0.0, prune_ratio=None, n_epoch=10)
    multi_history.append_history("vgg16 0", h)
    h = exec_vgg16(max_percent_per_iteration=0.2, prune_ratio=0.2, n_epoch=10)
    multi_history.append_history("vgg16 20", h)

    # h = exec_dense_net(max_percent_per_iteration=0.0, prune_ratio=None, n_epoch=10)
    # multi_history.append_history("densenet121 0", h)
    # h = exec_dense_net(max_percent_per_iteration=0.2, prune_ratio=0.2, n_epoch=10)
    # multi_history.append_history("densenet121 20", h)



    save_obj(multi_history, "history_compare")
    multi_history.display_single_key(History.VAL_ACC_KEY)


def run_alex_prune_compare():
    multi_history = MultiHistory()
    h = exec_alexnet(max_percent_per_iteration=0.0, prune_ratio=None, n_epoch=20)
    multi_history.append_history("Alexnet 0%", h)
    h = exec_alexnet(max_percent_per_iteration=0.05, prune_ratio=0.1)
    multi_history.append_history("Alexnet 10%-1", h)
    h = exec_alexnet(max_percent_per_iteration=0.15, prune_ratio=0.3)
    multi_history.append_history("Alexnet 30%-1", h)
    # h = exec_alexnet(max_percent_per_iteration=0.1, prune_ratio=0.3)
    # multi_history.append_history("Alexnet 30%-3", h)
    h = exec_alexnet(max_percent_per_iteration=0.25, prune_ratio=0.5)
    multi_history.append_history("Alexnet 50%-2", h)
    h = exec_alexnet(max_percent_per_iteration=0.25, prune_ratio=0.75)
    multi_history.append_history("Alexnet 75%-3", h)
    save_obj(multi_history, "history_alex")
    multi_history.display_single_key(History.VAL_ACC_KEY)


if __name__ == '__main__':
    multi_history = MultiHistory()
    # h = exec_vgg16(max_percent_per_iteration=0.3, prune_ratio=0.3, n_epoch=1)
    # multi_history.append_history("vgg16 20", h)
    # h = exec_dense_net(max_percent_per_iteration=0.3, prune_ratio=0.3, n_epoch=1)
    # multi_history.append_history("densenet121 20", h)

    run_alex_prune_compare()
    run_startegy_prune_compare()
