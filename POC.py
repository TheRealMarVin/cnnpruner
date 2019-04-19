import os
from heapq import nsmallest
from operator import itemgetter

import numpy as np
import torch
from deeplib.datasets import train_valid_loaders

from torch import nn
from torch.optim.lr_scheduler import  StepLR
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.models.resnet import BasicBlock
from torchvision.transforms import transforms, ToTensor

from CustomDeepLib import train, test, do_epoch, validate
from ExecutionGraphHelper import generate_graph, get__input_connection_count_per_entry
from FileHelper import load_obj, save_obj
from ModelHelper import get_node_in_model, total_num_filters
from models.AlexNetSki import alexnetski

# TODO check this one!!! https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
###
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
        self.connection_count = {}
        self.features = []
        self.reset()
        model.cpu()
        self.graph, self.name_dic, self.root = generate_graph(model, sample_run)

        # get__input_connection_count_per_entry(self.graph, self.root, self.connection_count)
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

            # if node_id == "246":
            #     a = 0
            # out = curr_module(x)
            # print("\t module name: {} \tbefore shape: {}\tafter shape:{}".format(node_name, x.shape, out.shape))
            # if node_id == "232":
            #     a = 0

            if isinstance(curr_module, torch.nn.modules.conv.Conv2d):
                # print("name: {}\t\tnode_id: {}\tactivation_index: {}".format(node_name, node_id, self.activation_index))
                # out.register_hook(self.compute_rank)
                self.conv_layer[node_id] = curr_module
                # self.activations[node_id] = out
                # self.activation_to_layer[self.activation_index] = node_id
                # self.activation_index += 1
                self.hook = curr_module.register_forward_hook(self.hook_fn)

            out = curr_module(x)

        res = None
        next_nodes = self.graph[node_id]
        if len(next_nodes) == 0:
            res = out
        else:
            # execute next
            for next_id in self.graph[node_id].split(","):
                # next_id
                self.connection_count[next_id] -= 1
                if next_id in self.forward_res:
                    # print("AAA adding weights")
                    # self.forward_res[next_id].require_grad = True
                    self.forward_res[next_id] = self.forward_res[next_id] + out
                else:
                    self.forward_res[next_id] = out

                res = self.parse(next_id)
        return res

    def hook_fn(self, module, input, output):
        self.features.append(torch.tensor(output, requires_grad=True).cuda())
    #
    # def close(self):
    #     self.hook.remove()

    # This is super slow because of the way I parse the execution tree, but it works
    def forward(self, x):
        self.activations = {}
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}
        self.forward_res = {}
        self.conv_layer = {}

        self.activation_index = 0

        get__input_connection_count_per_entry(self.graph, self.root, self.connection_count)
        self.layer_to_parse = self.graph.keys()

        self.connection_count[self.root] = 0    # for the root we have everything we need
        self.forward_res[self.root] = x         # for root we also have the proper input

        x.requires_grad = True
        x = self.parse(self.root)
        return x

    def extract_grad(self, out):
        with torch.no_grad():
            for node_name, curr_module in self.conv_layer.items():
                # curr_module = self.conv_layer[node_name]
                grad = curr_module.weight.grad
                # means = [x.view(-1).mean() for x in grad]
                activation = curr_module.weight
                pdist = nn.PairwiseDistance(p=2)
                out = pdist(activation, grad)
                # o1 = torch.abs(out)
                # o1 = o1 / torch.sqrt(torch.sum(o1 * v))
                means2 = torch.tensor([x.view(-1).mean() for x in out]).cuda() #TODO I think it should be negative
                if node_name not in self.filter_ranks:
                    self.filter_ranks[node_name] = means2
                else:
                    self.filter_ranks[node_name] = self.filter_ranks[node_name] + means2


    # def compute_rank(self, grad):
    #     activation_index = len(self.activations) - self.grad_index - 1
    #     print("grad_index: {} \tactivation index: {}".format(self.grad_index, activation_index))
    #     if grad._backward_hooks is not None:
    #         a = 0
    #     ###
    #     # atl = self.activation_to_layer[activation_index]
    #     # print("Activation index: {} \tcorrespond to layer:{}".format(activation_index, atl))
    #     ###
    #     activation = self.activations[activation_index]
    #     # values = torch.sum((activation * grad), dim = 2).sum(dim=2).sum(dim=3)[0, :, 0, 0].data
    #
    #     # normalized = torch.mul(ag_dot[1], 1 / (activation.size(0) * activation.size(2) * activation.size(3)))
    #
    #     # Normalize the rank by the filter dimensions
    #     # values = values / (activation.size(0) * activation.size(2) * activation.size(3))
    #     if activation_index not in self.filter_ranks:
    #         self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_().cuda()
    #         print("ninja ai:{}".format(activation_index))
    #     print("tortue")
    #
    #     ag_dot = activation * grad
    #     normalized = torch.mul(torch.sum(ag_dot, dim=2).sum(dim=2),
    #                            1 / (activation.size(0) * activation.size(2) * activation.size(3)))
    #     for i in range(normalized.size(0)):
    #         self.filter_ranks[activation_index] += normalized[i]
    #     self.grad_index += 1

    def sort_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((i, j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / torch.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v

    def plan_prunning(self, num_filters_to_prune):
        # aaa = nsmallest(num_filters_to_prune, self.filter_ranks, itemgetter(2))
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
            print("trying to prune for layer: {} \tID: {}".format(self.name_dic[layer_id], layer_id))
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
                        else:
                            print("already affected by pruning")
                    # self._apply_pruning_effect(layer_id, filters_to_remove, initial_filter_count, effect_applied)

    def _apply_pruning_effect(self, layer_id, removed_filter, initial_filter_count, effect_applied):
        # next_id = self.graph[layer_id]
        if layer_id not in self.name_dic:
            for sub_node_id in layer_id.split(","):
                self._apply_pruning_effect(sub_node_id, removed_filter, initial_filter_count)
            print("end of effect after loop")
            return
        layer = get_node_in_model(self.model, self.name_dic[layer_id])
        print("\tapply pruning effect on: {} \tID: {}".format(self.name_dic[layer_id], layer_id))
        if layer_id == "211":
            a = 0

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
                else:
                    print("already affected by pruning")
        else:
            print("\t\t{} has no more".format(self.name_dic[layer_id]))

    def _prune_conv_output_filters(self, conv, filters_to_remove):
        # TODO try not using cpu
        initial_filter_count = conv.out_channels
        conv.out_channels = conv.out_channels - len(filters_to_remove)
        old_weights = conv.weight.data.cpu().detach()
        # TODO make sure there is no overflow
        new_weights = np.delete(old_weights, filters_to_remove, 0)
        conv.weight.data = new_weights.cuda()
        # print("conv  _ our _ old weight shape {} vs new weight shape {}".format(old_weights.shape, new_weights.shape))
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
        # weight scaling because removing nodes is basically like a form of dropout
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


# #TODO remove
# def find_layer_and_next(module, layer_name, in_desired_layer=None):
#     desired_layer = in_desired_layer
#     next_desired_layer = None
#
#     splitted = layer_name.split(".")
#     if len(splitted) > 0:
#         sub = splitted[1:]
#         next_layer_name = '.'.join(sub)
#     elif len(splitted) == 0:
#         next_layer_name = splitted[0]
#
#     for name, curr_module in module.named_children():
#         if desired_layer is None \
#                 and name == layer_name \
#                 and isinstance(curr_module, torch.nn.modules.conv.Conv2d):
#             desired_layer = curr_module
#         else:
#             if desired_layer is not None:
#                 if isinstance(curr_module, torch.nn.modules.conv.Conv2d):
#                     next_desired_layer = curr_module
#                     break
#                 elif isinstance(curr_module, torch.nn.modules.Linear):
#                     next_desired_layer = curr_module
#                     break
#                 elif isinstance(curr_module, torch.nn.modules.BatchNorm2d):
#                     next_desired_layer = curr_module
#                     break
#             res_desired, res_next= find_layer_and_next(curr_module, next_layer_name, desired_layer)
#             if desired_layer is None and res_desired is not None:
#                 desired_layer = res_desired
#             if next_desired_layer is None and res_next is not None:
#                 next_desired_layer = res_next
#
#             if desired_layer is not None and next_desired_layer is not None:
#                 return desired_layer, next_desired_layer
#     return desired_layer, next_desired_layer
#
# #TODO remove
# #TODO on devrait les faire en batch ca irait pas mal plus vite
# def prune(model, layer_index, filter_index):
#     conv, next_layer = find_layer_and_next(model, layer_index)
#
#     # TODO try not using cpu
#     conv.out_channels = conv.out_channels - 1
#     old_weights = conv.weight.data.cpu().detach()
#     new_weights = np.delete(old_weights, [filter_index], 0)
#     conv.weight.data = new_weights.cuda()
#     print("old weight shape {} vs new weight shape {}".format(old_weights.shape, new_weights.shape))
#     conv.weight._grad = None
#
#     if conv.bias is not None:
#         # TODO try not using cpu
#         bias_numpy = conv.bias.data.cpu().detach()
#         new_bias_numpy = np.delete(bias_numpy, [filter_index], 0)
#         conv.bias.data = new_bias_numpy.cuda()
#         conv.bias._grad = None
#
#     if isinstance(next_layer, torch.nn.modules.conv.Conv2d):
#         # TODO try not using cpu
#         next_layer.in_channels = next_layer.in_channels - 1
#         old_weights = next_layer.weight.data.cpu()
#         new_weights = np.delete(old_weights, [filter_index], 1)
#         next_layer.weight.data = new_weights.cuda()
#         print("old weight shape {} vs new weight shape {}".format(old_weights.shape, new_weights.shape))
#         next_layer.weight._grad = None
#
#     elif isinstance(next_layer, torch.nn.modules.Linear):
#         lin_in_feat = next_layer.in_features
#         conv_out_channels = conv.out_channels
#
#         elem_per_channel = (lin_in_feat//conv_out_channels)
#         new_lin_in_feat = lin_in_feat - elem_per_channel
#         old_lin_weights = next_layer.weight.detach()
#         lin_new_weigths = np.delete(old_lin_weights, [x + filter_index * elem_per_channel for x in range(elem_per_channel)], 1)
#         #weight scaling because removing nodes is basically like a form of dropout
#         factor = 1 - (elem_per_channel / lin_in_feat)
#         lin_new_weigths.mul_(factor)
#         next_layer.weight.data = lin_new_weigths
#         next_layer.in_features = new_lin_in_feat
#         next_layer.weight._grad = None
#
#     elif isinstance(next_layer, torch.nn.modules.BatchNorm2d):
#         # print("nb features: ", next_layer.num_features)
#         #TODO on network that doesn't converge it could reach 0... at this point we might want to remove it completely... maybe
#         if next_layer.num_features > 10:
#             next_layer.num_features = next_layer.num_features - 1
#             old_batch_weights = next_layer.weight.detach()
#             new_batch_weights = np.delete(old_batch_weights, [filter_index], 0)
#             next_layer.weight.data = new_batch_weights
#
#             if next_layer.bias is not None:
#                 # TODO try not using cpu
#                 bias_numpy = next_layer.bias.data.cpu().detach()
#                 new_bn_bias_numpy = np.delete(bias_numpy, [filter_index], 0)
#                 next_layer.bias.data = new_bn_bias_numpy.cuda()
#                 next_layer.bias._grad = None
#
#             next_layer.weight._grad = None
#             if next_layer.track_running_stats:
#                 next_layer.register_buffer('running_mean', torch.zeros(next_layer.num_features))
#                 next_layer.register_buffer('running_var', torch.ones(next_layer.num_features))
#                 next_layer.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
#             else:
#                 next_layer.register_parameter('running_mean', None)
#                 next_layer.register_parameter('running_var', None)
#                 next_layer.register_parameter('num_batches_tracked', None)
#             next_layer.reset_running_stats()
#             next_layer.reset_parameters()
#
#     return model
# ###


# def total_num_filters(modules):
#     filters = 0
#
#     if isinstance(modules, torch.nn.modules.conv.Conv2d):
#         filters = filters + modules.out_channels
#     else:
#         if len(modules._modules.items()) > 0:
#             for name, sub_module in modules._modules.items():
#                 if sub_module is not None:
#                     filters = filters + total_num_filters(sub_module)
#
#         else:
#             if isinstance(modules, torch.nn.modules.conv.Conv2d):
#                 filters = filters + modules.out_channels
#     return filters
###


def common_training_code(model, pruned_save_path=None,
                         best_result_save_path=None, retrain_if_weight_loaded=False,
                         sample_run=None,
                         reuse_cut_filter=False):
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
    batch_size = 128

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
    num_filters_to_prune_per_iteration = 512 #TODO calculer automatiquement
    # iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
    ratio = 1.0/3
    iterations = int(number_of_filters * ratio)//num_filters_to_prune_per_iteration
    print("{} iterations to reduce {:2.2f}% filters".format(iterations, ratio*100))

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
            train(model, optimizer, train_dataset, 1, 64, use_gpu=use_gpu, criterion=criterion,
                  scheduler=scheduler, pruner=pruner, batch_count=1, should_validate=False)
            # if train_dataset.transform is None:
            #     train_dataset.transform = ToTensor()
            #
            # train_loader, val_loader = train_valid_loaders(train_dataset, batch_size=batch_size)
            # # do_epoch(criterion, model, optimizer, scheduler, train_loader, use_gpu, pruner=pruner, count=4)
            # validate(model, train_loader, use_gpu=True, pruner=None)

            pruner.normalize_layer()

            prune_targets = pruner.plan_prunning(num_filters_to_prune_per_iteration)
            if reuse_cut_filter:
                save_obj(prune_targets, "filters_dic")

        pruner.display_pruning_log(prune_targets)

        print("Pruning filters.. ")
        model = model.cpu()
        pruner.prune(prune_targets)
        # for layer_index, filter_index in prune_targets:
        #     model = prune(model, layer_index, filter_index)

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
              scheduler=scheduler, prunner=None)
        history.display()

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
    exec_q3b()
    # exec_poc()
    # exec_poc2()


if __name__ == '__main__':
    exec_q3()
