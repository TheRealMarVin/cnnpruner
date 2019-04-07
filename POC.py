import os
import random
from heapq import nsmallest
from operator import itemgetter
from shutil import copyfile

import numpy as np
import torch
import time

from torch.utils import model_zoo
from torchvision.models.alexnet import model_urls

from deeplib.datasets import train_valid_loaders
from deeplib.history import History
from deeplib.training import do_epoch, validate
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import  StepLR
from torch.utils.data import SequentialSampler
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms, ToTensor

from models.AlexNetSki import alexnetski


def train(model, optimizer, dataset, n_epoch, batch_size, use_gpu=True, scheduler=None,
          criterion=None, prunner=None, retain_graph=None, best_result_save_path=None):
    history = History()

    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    """
    if someone provides a transform upstream.
     there is chances this person what the transform and not silently override it.
    """
    if dataset.transform is None:
        dataset.transform = ToTensor()

    train_loader, val_loader = train_valid_loaders(dataset, batch_size=batch_size)

    highest_score = 0.0
    for i in range(n_epoch):
        start = time.time()
        do_epoch(criterion, model, optimizer, scheduler, train_loader, use_gpu, prunner=prunner, retain_graph=retain_graph)
        end = time.time()

        train_acc, train_loss = validate(model, train_loader, use_gpu)
        val_acc, val_loss = validate(model, val_loader, use_gpu)
        history.save(train_acc, val_acc, train_loss, val_loss, optimizer.param_groups[0]['lr'])

        if best_result_save_path is not None \
                and val_acc > highest_score:
            highest_score = val_acc
            if os.path.isfile(best_result_save_path):
                copyfile(best_result_save_path, best_result_save_path + ".old")

            basedir = os.path.dirname(best_result_save_path)
            if not os.path.exists(basedir):
                os.makedirs(basedir)
            torch.save(model.state_dict(), best_result_save_path)
        print(
            'Epoch {} - Train acc: {:.2f} - Val acc: {:.2f} - Train loss: {:.4f} - Val loss: {:.4f} - Training time: {:.2f}s'.format(
                i, train_acc, val_acc, train_loss, val_loss, end - start))

    return history


# fonction de test qui n'override pas la transform
def test(model, test_dataset, batch_size, use_gpu=True):
    sampler = SequentialSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=sampler)

    score, loss = validate(model, test_loader, use_gpu=use_gpu)
    return score


def plot_images(images, cls_true, label_names=None, cls_pred=None, score=None, gray=False):
    assert len(images) == len(cls_true) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        if gray:
            ax.imshow(images[i], cmap='gray', interpolation='spline16')
        else:
            ax.imshow(images[i], interpolation='spline16')
        # get its equivalent class name

        if label_names:
            cls_true_name = label_names[cls_true[i]]

            if cls_pred is None:
                xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
            elif score is None:
                cls_pred_name = label_names[cls_pred[i]]
                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
            else:
                cls_pred_name = label_names[cls_pred[i]]
                xlabel = "True: {0}\nPred: {1}\nScore: {2:.2f}%".format(cls_true_name, cls_pred_name, score[i] * 100)

            ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def display_sample_data(train_dataset):
    idx = random.sample([x for x in range(len(train_dataset))], 9)
    images = [np.array(train_dataset[i][0]).squeeze().transpose((1, 2, 0)) for i in idx]
    targets = [train_dataset[i][1] for i in idx]

    plot_images(images, targets, gray=False)

###
class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}
        self.filter_ranks = {}
        self.activation_index = 0

    def _register_hook(self, x, module, layer):
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            x.requires_grad = True
            x.register_hook(self.compute_rank)
            self.activations.append(x)
            self.activation_to_layer[self.activation_index] = layer
            self.activation_index += 1
        else:
            # TODO je peux surement éviter d'utiliser le _modules... ou pas
            if len(module._modules.items()) > 0:
                for sub_layer, sub_module in module._modules.items():
                    if sub_module is not None:
                        desired_layer = sub_layer
                        if len(layer) > 0:
                            desired_layer = layer + "." + desired_layer
                        self._register_hook(x, sub_module, desired_layer)


    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        self.activation_index = 0

        #TODO sacrer ca dans la fonction c'est laid!
        for name, module in self.model._modules.items():
            self._register_hook(x, module, name)
        x = self.model(x)

        return x

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        # values = torch.sum((activation * grad), dim = 2).sum(dim=2).sum(dim=3)[0, :, 0, 0].data
        ag_dot = activation * grad
        normalized = torch.mul(ag_dot[1], 1 / (activation.size(0) * activation.size(2) * activation.size(3)))

        # Normalize the rank by the filter dimensions
        # values = values / (activation.size(0) * activation.size(2) * activation.size(3))

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_().cuda()

        normalized = torch.mul(torch.sum(ag_dot, dim=2).sum(dim=2),
                               1 / (activation.size(0) * activation.size(2) * activation.size(3)))
        for i in range(normalized.size(0)):
            self.filter_ranks[activation_index] += normalized[i]
        self.grad_index += 1

    def sort_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / torch.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v

    #TODO ca c'est pas bon, c'est inspiré d'un truc trouvé sur le web, mais moi je veux tous les indices dans le bon ordre
    def plan_prunning(self, num_filters_to_prune):
        filters_to_prune = self.sort_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune
###



def find_layer_and_next(module, layer_name, in_desired_layer=None):
    desired_layer = in_desired_layer
    next_desired_layer = None

    splitted = layer_name.split(".")
    if len(splitted) > 0:
        sub = splitted[1:]
        next_layer_name = '.'.join(sub)
    elif len(splitted) == 0:
        next_layer_name = splitted[0]

    for name, curr_module in module.named_children():
        if desired_layer is None \
                and name == layer_name \
                and isinstance(curr_module, torch.nn.modules.conv.Conv2d):
            desired_layer = curr_module
        else:
            if desired_layer is not None:
                if isinstance(curr_module, torch.nn.modules.conv.Conv2d):
                    next_desired_layer = curr_module
                    break
                elif isinstance(curr_module, torch.nn.modules.Linear):
                    next_desired_layer = curr_module
                    break
                elif isinstance(curr_module, torch.nn.modules.BatchNorm2d):
                    next_desired_layer = curr_module
                    break
            res_desired, res_next= find_layer_and_next(curr_module, next_layer_name, desired_layer)
            if desired_layer is None and res_desired is not None:
                desired_layer = res_desired
            if next_desired_layer is None and res_next is not None:
                next_desired_layer = res_next

            if desired_layer is not None and next_desired_layer is not None:
                return desired_layer, next_desired_layer
    return desired_layer, next_desired_layer


#TODO on devrait les faire en batch ca irait pas mal plus vite
def prune(model, layer_index, filter_index):
    conv, next_layer = find_layer_and_next(model, layer_index)

    conv.out_channels = conv.out_channels - 1
    old_weights = conv.weight.data.cpu().detach()
    new_weights = np.delete(old_weights, [filter_index], 0)
    conv.weight.data = new_weights.cuda()
    conv.weight._grad = None

    if conv.bias is not None:
        bias_numpy = conv.bias.data.cpu().detach()
        new_bias_numpy = np.delete(bias_numpy, [filter_index], 0)
        conv.bias.data = new_bias_numpy.cuda()
        conv.bias._grad = None

    if isinstance(next_layer, torch.nn.modules.conv.Conv2d):
        next_layer.in_channels = next_layer.in_channels - 1
        old_weights = next_layer.weight.data.cpu()
        new_weights = np.delete(old_weights, [filter_index], 1)
        next_layer.weight.data = new_weights.cuda()
        next_layer.weight._grad = None

    elif isinstance(next_layer, torch.nn.modules.Linear):
        lin_in_feat = next_layer.in_features
        conv_out_channels = conv.out_channels

        elem_per_channel = (lin_in_feat//conv_out_channels)
        new_lin_in_feat = lin_in_feat - elem_per_channel
        old_lin_weights = next_layer.weight.detach()
        lin_new_weigths = np.delete(old_lin_weights, [x + filter_index * elem_per_channel for x in range(elem_per_channel)], 1)
        #weight scaling because removing nodes is basically like a form of dropout
        factor = 1 - (elem_per_channel / lin_in_feat)
        lin_new_weigths.mul_(factor)
        next_layer.weight.data = lin_new_weigths
        next_layer.in_features = new_lin_in_feat
        next_layer.weight._grad = None

    elif isinstance(next_layer, torch.nn.modules.BatchNorm2d):
        # print("nb features: ", next_layer.num_features)
        #TODO on network that doesn't converge it could reach 0... at this point we might want to remove it completely... maybe
        if next_layer.num_features > 10:
            next_layer.num_features = next_layer.num_features - 1
            old_batch_weights = next_layer.weight.detach()
            new_batch_weights = np.delete(old_batch_weights, [filter_index], 0)
            next_layer.weight.data = new_batch_weights
            next_layer.weight._grad = None
            if next_layer.track_running_stats:
                next_layer.register_buffer('running_mean', torch.zeros(next_layer.num_features))
                next_layer.register_buffer('running_var', torch.ones(next_layer.num_features))
                next_layer.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
            else:
                next_layer.register_parameter('running_mean', None)
                next_layer.register_parameter('running_var', None)
                next_layer.register_parameter('num_batches_tracked', None)
            next_layer.reset_running_stats()
            next_layer.reset_parameters()

    return model
###


def total_num_filters(modules):
    filters = 0

    if isinstance(modules, torch.nn.modules.conv.Conv2d):
        filters = filters + modules.out_channels
    else:
        if len(modules._modules.items()) > 0:
            for name, sub_module in modules._modules.items():
                if sub_module is not None:
                    filters = filters + total_num_filters(sub_module)

        else:
            if isinstance(modules, torch.nn.modules.conv.Conv2d):
                filters = filters + modules.out_channels
    return filters
###


def common_code_for_q3(model, pruned_save_path=None, best_result_save_path=None, retrain_if_weight_loaded=False):
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
    n_epoch_retrain = 2
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
    pruner = FilterPrunner(model)
    number_of_filters = total_num_filters(model)
    num_filters_to_prune_per_iteration = 256
    iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
    ratio = 2.0/3
    iterations = int(iterations * ratio)
    print("{} iterations to reduce {:2.2f}% filters".format(iterations, ratio*100))

    for param in model.parameters():
        param.requires_grad = True

    for iteration_idx in range(iterations):
        print("Perform pruning iteration: {}".format(iteration_idx))
        pruner.reset()

        train(model, optimizer, train_dataset, 1, batch_size, use_gpu=use_gpu, criterion=criterion,
              scheduler=scheduler, prunner=pruner)

        pruner.normalize_layer()

        prune_targets = pruner.plan_prunning(num_filters_to_prune_per_iteration)
        layers_prunned = {}
        for layer_index, filter_index in prune_targets:
            if layer_index not in layers_prunned:
                layers_prunned[layer_index] = 0
            layers_prunned[layer_index] = layers_prunned[layer_index] + 1

        print("Layers that will be pruned", layers_prunned)
        print("Pruning filters.. ")
        model = model.cpu()
        for layer_index, filter_index in prune_targets:
            model = prune(model, layer_index, filter_index)

        model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.007)
        pruner.reset()

        print("Filters pruned {}%".format(100 * float(total_num_filters(model)) / number_of_filters))
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

    common_code_for_q3(model, pruned_save_path="../saved/alex/PrunedAlexnet.pth",
                       best_result_save_path="../saved/alex/alexnet.pth")


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

    common_code_for_q3(model, pruned_save_path="../saved/resnet/Prunedresnet.pth,",
                       best_result_save_path="../saved/resnet/resnet18.pth")


def exec_q3():
    exec_q3b()
    # exec_poc()


if __name__ == '__main__':
    exec_q3()
