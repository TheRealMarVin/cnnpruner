import copy
import os
import random
from heapq import nsmallest, nlargest
from operator import itemgetter

import numpy as np
import torch

from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.models.resnet import BasicBlock
from torchvision.transforms import transforms

from CustomDeepLib import train, test
from ExecutionGraphHelper import generate_graph, get_input_connection_count_per_entry
from FileHelper import load_obj, save_obj
from ModelHelper import get_node_in_model, total_num_filters
from models.AlexNetSki import alexnetski

# TODO check this one!!! https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
# and this: https://github.com/fg91/visualizing-cnn-feature-maps/blob/master/Calculate_mean_activation_per_filter_in_specific_layer_given_an_image.ipynb
from models.FResiNet import FResiNet


class FilterPruner:
    def __init__(self, model, sample_run):
        self.model = model
        self.activations = {}
        self.gradients = []
        self.grad_index = 0 # TODO remove
        self.conv_layer = {}
        self.activation_to_layer = {} # TODO remove
        self.filter_ranks = {}
        self.forward_res = {}
        self.activation_index = 0 # TODO remove
        self.test_layer_activation = {} #TODO tata
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
                # means = torch.tensor([curr.view(-1).mean() for curr in out]).cuda()
                average_per_batch_item = torch.tensor([[curr.view(-1).mean() for curr in batch_item] for batch_item in out])
                activation_average_sum = torch.sum(average_per_batch_item, dim=0)

                val = activation_average_sum.cuda()
                # means = torch.tensor([out[0,i].mean().item() for i in range(out.shape[0])], requires_grad=True).cuda() #TODO not sure I need grad at all here!
                # self.test_layer_activation[node_id] = means
                if node_id not in self.test_layer_activation:
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

    #TODO most probably useless
    def extract_filter_activation_mean(self, out):
        with torch.no_grad():
            for node_name, curr_module in self.conv_layer.items():
                if self.is_before_merge(node_name):
                    continue

                # activations = self.test_layer_activation[node_name]
                # mean_act = [activations.features[0, i].mean().item() for i in range(total_filters_in_layer)]
                # grad = curr_module.weight.grad
                # activation = curr_module.weight
                # pdist = nn.PairwiseDistance(p=2)
                # out = pdist(activation, grad)
                # means2 = torch.tensor([x.view(-1).mean() for x in out]).cuda() #TODO I think it should be negative
                #
                # pouet = self.test_layer_activation[node_name]
                # diff = activation - pouet
                # means4 = torch.tensor([x.view(-1).mean() for x in diff]).cuda()
                #
                # out2 = pdist(pouet, grad)
                # means3 = torch.tensor([x.view(-1).sum() for x in grad]).cuda()  # TODO I think it should be negative

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
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])

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
        # TODO try not using cpu
        initial_filter_count = conv.out_channels
        conv.out_channels = conv.out_channels - len(filters_to_remove)
        old_weights = conv.weight.data.cpu().detach()
        # TODO make sure there is no overflow
        new_weights = np.delete(old_weights, filters_to_remove, 0)
        conv.weight.data = new_weights.cuda()
        conv.weight._grad = None

        if conv.bias is not None:
            # TODO try not using cpu
            bias_numpy = conv.bias.data.cpu().detach()
            # TODO make sure there is no overflow
            new_bias_numpy = np.delete(bias_numpy, filters_to_remove, 0)
            conv.bias.data = new_bias_numpy.cuda()
            conv.bias._grad = None

        return initial_filter_count

    def _prune_conv_input_filters(self, conv, removed_filter, _):
        # TODO try not using cpu
        conv.in_channels = conv.in_channels - len(removed_filter)
        old_weights = conv.weight.data.cpu()
        # TODO make sure there is no overflow
        new_weights = np.delete(old_weights, removed_filter, 1)
        conv.weight.data = new_weights.cuda()
        # print("conc _ in _ old weight shape {} vs new weight shape {}".format(old_weights.shape, new_weights.shape))
        conv.weight._grad = None

    def _prune_conv_input_batchnorm(self, batchnorm, removed_filter, _):
        batchnorm.num_features = batchnorm.num_features - len(removed_filter)
        old_batch_weights = batchnorm.weight.detach()
        new_batch_weights = np.delete(old_batch_weights, removed_filter, 0)
        batchnorm.weight.data = new_batch_weights

        if batchnorm.bias is not None:
            # TODO try not using cpu
            bias_numpy = batchnorm.bias.data.cpu().detach()
            new_bn_bias_numpy = np.delete(bias_numpy, removed_filter, 0)
            batchnorm.bias.data = new_bn_bias_numpy.cuda()
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

def common_training_code(model, pruned_save_path=None,
                         best_result_save_path=None, retrain_if_weight_loaded=False,
                         sample_run=None,
                         reuse_cut_filter=False,
                         max_percent_per_iteration=0.1,
                         prune_ratio=0.3):
    test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = CIFAR10("C:/dev/data/cifar10/", train=True, transform=test_transform, download=True)
    test_dataset = CIFAR10("C:/dev/data/cifar10/", train=False, transform=test_transform, download=True)

    # display_sample_data(train_dataset)
    # display_sample_data(test_dataset)
    model.cuda()

    use_gpu = True
    n_epoch = 1
    n_epoch_retrain = 1
    batch_size = 128 # TODO use 128

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.007)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
    #
    should_train = True
    if best_result_save_path is not None:
        if os.path.isfile(best_result_save_path):
            model.load_state_dict(torch.load(best_result_save_path))
            if not retrain_if_weight_loaded:
                should_train = False
    if should_train:
        history = train(model, optimizer, train_dataset, n_epoch,
                        batch_size, use_gpu=use_gpu, criterion=criterion,
                        scheduler=scheduler, best_result_save_path=best_result_save_path)
        history.display()

    test_score = test(model, test_dataset, batch_size, use_gpu=use_gpu)
    print('Test:\n\tScore: {}'.format(test_score))

    ###
    pruner = FilterPruner(model, sample_run)
    number_of_filters = total_num_filters(model)
    filter_to_prune = (int)(number_of_filters * prune_ratio)
    max_filters_to_prune_on_iteration = (int)(number_of_filters * max_percent_per_iteration)
    if filter_to_prune < max_filters_to_prune_on_iteration:
        max_filters_to_prune_on_iteration = filter_to_prune
    iterations = (filter_to_prune//max_filters_to_prune_on_iteration) + 1
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
            #TODO should use less batch in an epoch and bigger batch size
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
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.007)
        pruner.reset()

        print("Filters pruned {}%".format(100 - (100 * float(total_num_filters(model)) / number_of_filters)))
        new_test_score = test(model, test_dataset, batch_size, use_gpu=use_gpu)
        print('Test:\n\tpost prune Score: {}'.format(new_test_score))

        basedir = os.path.dirname(pruned_save_path)
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        torch.save(model, pruned_save_path)

        print("Fine tuning to recover from prunning iteration.")
        history = train(model, optimizer, train_dataset, n_epoch_retrain, batch_size, use_gpu=use_gpu, criterion=None,
              scheduler=scheduler, pruner=None)
        history.display()
        test_score = test(model, test_dataset, batch_size, use_gpu=use_gpu)
        print('Test pruning iteration :{}\n\tScore: {}'.format(iteration_idx, test_score))

    ###

    test_score = test(model, test_dataset, batch_size, use_gpu=use_gpu)
    print('Test Fin :\n\tScore: {}'.format(test_score))


def exec_poc():
    print("Proof of concept")
    model = alexnetski(pretrained=True)
    model.cuda()

    # TODO reuse_cut_filter must be false
    common_training_code(model, pruned_save_path="../saved/alex/PrunedAlexnet.pth",
                         best_result_save_path="../saved/alex/alexnet.pth",
                         sample_run=torch.zeros([1, 3, 224, 224]),
                         reuse_cut_filter=False)

def exec_poc2():
    print("Proof of concept")
    model = FResiNet(BasicBlock, [2, 2, 2])
    model.cuda()

    # TODO reuse_cut_filter must be false
    common_training_code(model, pruned_save_path="../saved/fresinet/PrunedFresinet.pth",
                         # best_result_save_path=None,
                         best_result_save_path="../saved/fresinet/fresinet.pth",
                         sample_run=torch.zeros([1, 3, 224, 224]),
                         reuse_cut_filter=True)


def exec_q3b():
    print("question 3b")

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model.conv1.requires_grad = False
    model.bn1.requires_grad = False
    model.relu.requires_grad = False
    model.maxpool.requires_grad = False
    model.layer1.requires_grad = False
    model.layer2.requires_grad = False
    model.layer3.requires_grad = False
    model.layer4.requires_grad = False

    model.cuda()

    #TODO reuse_cut_filter must be false
    common_training_code(model, pruned_save_path="../saved/resnet/Prunedresnet.pth,",
                         best_result_save_path="../saved/resnet/resnet18.pth",
                         sample_run=torch.zeros([1, 3, 224, 224]),
                         reuse_cut_filter=False)


def exec_q3():
    # exec_q3b()
    exec_poc()
    # exec_poc2()


if __name__ == '__main__':
    exec_q3()
