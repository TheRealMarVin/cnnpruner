import os

import torch
from torch import nn
from torchvision import models

from POC import common_training_code
from models.AlexNetSki import alexnetski


def exec_alexnet(exec_name, pruning_params=None, exec_params=None, dataset_params=None):
    print("*** ", exec_name)
    if exec_params.best_result_save_path is not None and os.path.isfile(exec_params.best_result_save_path):
        model = torch.load(exec_params.best_result_save_path)
    else:
        model = alexnetski(pretrained=True)

    model.cuda()

    history, test_score = common_training_code(model,
                                               pruned_save_path="../saved/{}/Pruned.pth".format(exec_name),
                                               pruned_best_result_save_path="../saved/{}/pruned_best.pth".format(exec_name),
                                               sample_run=torch.zeros([1, 3, 224, 224]),
                                               pruning_params=pruning_params,
                                               exec_params=exec_params,
                                               dataset_params=dataset_params)

    return history


def exec_resnet18(exec_name, pruning_params=None, exec_params=None, dataset_params=None, out_count=1000):
    print("*** ", exec_name)

    if exec_params.best_result_save_path is not None and os.path.isfile(exec_params.best_result_save_path):
        model = torch.load(exec_params.best_result_save_path)
    else:
        model = models.resnet18(pretrained=True)
        if model.fc.out_features != out_count:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 10)

    model.cuda()

    history, test_score = common_training_code(model,
                                               pruned_save_path="../saved/{}/Pruned.pth".format(exec_name),
                                               pruned_best_result_save_path="../saved/{}/pruned_best.pth".format(exec_name),
                                               sample_run=torch.zeros([1, 3, 224, 224]),
                                               pruning_params=pruning_params,
                                               exec_params=exec_params,
                                               dataset_params=dataset_params)
    return history


def exec_resnet34(exec_name, pruning_params=None, exec_params=None, dataset_params=None, out_count=1000):
    print("*** ", exec_name)

    if exec_params.best_result_save_path is not None and os.path.isfile(exec_params.best_result_save_path):
        model = torch.load(exec_params.best_result_save_path)
    else:
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)

    model.cuda()

    history, test_score = common_training_code(model,
                                               pruned_save_path="../saved/{}/Pruned.pth".format(exec_name),
                                               pruned_best_result_save_path="../saved/{}/pruned_best.pth".format(exec_name),
                                               sample_run=torch.zeros([1, 3, 224, 224]),
                                               pruning_params=pruning_params,
                                               exec_params=exec_params,
                                               dataset_params=dataset_params)
    return history


def exec_resnet50(exec_name, pruning_params=None, exec_params=None, dataset_params=None, out_count=1000):
    print("*** ", exec_name)

    if exec_params.best_result_save_path is not None and os.path.isfile(exec_params.best_result_save_path):
        model = torch.load(exec_params.best_result_save_path)
    else:
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)

    model.cuda()

    history, test_score = common_training_code(model,
                                               pruned_save_path="../saved/{}/Pruned.pth".format(exec_name),
                                               pruned_best_result_save_path="../saved/{}/pruned_best.pth".format(exec_name),
                                               sample_run=torch.zeros([1, 3, 224, 224]),
                                               pruning_params=pruning_params,
                                               exec_params=exec_params,
                                               dataset_params=dataset_params)
    return history


def exec_squeeze_net(exec_name, pruning_params=None, exec_params=None, dataset_params=None):
    print("*** ", exec_name)

    if exec_params.best_result_save_path is not None and os.path.isfile(exec_params.best_result_save_path):
        model = torch.load(exec_params.best_result_save_path)
    else:
        model = models.squeezenet1_1(pretrained=True)

    model.cuda()

    if exec_params is not None:
        exec_params.force_forward_view = True
        exec_params.ignore_last_conv = True

    history, test_score = common_training_code(model,
                                               pruned_save_path="../saved/{}/Pruned.pth".format(exec_name),
                                               pruned_best_result_save_path="../saved/{}/pruned_best.pth".format(exec_name),
                                               sample_run=torch.zeros([1, 3, 224, 224]),
                                               pruning_params=pruning_params,
                                               exec_params=exec_params,
                                               dataset_params=dataset_params)
    return history


def exec_dense_net(exec_name, pruning_params=None, exec_params=None, dataset_params=None):
    print("*** ", exec_name)

    if exec_params.best_result_save_path is not None and os.path.isfile(exec_params.best_result_save_path):
        model = torch.load(exec_params.best_result_save_path)
    else:
        model = models.densenet121(pretrained=True)

    model.cuda()

    history, test_score = common_training_code(model,
                                               pruned_save_path="../saved/{}/Pruned.pth".format(exec_name),
                                               pruned_best_result_save_path="../saved/{}/pruned_best.pth".format(exec_name),
                                               sample_run=torch.zeros([1, 3, 224, 224]),
                                               pruning_params=pruning_params,
                                               exec_params=exec_params,
                                               dataset_params=dataset_params)
    return history


def exec_vgg16(exec_name, pruning_params=None, exec_params=None, dataset_params=None):
    print("*** ", exec_name)

    if exec_params.best_result_save_path is not None and os.path.isfile(exec_params.best_result_save_path):
        model = torch.load(exec_params.best_result_save_path)
    else:
        model = models.vgg16(pretrained=True)

    model.cuda()

    history, test_score = common_training_code(model,
                                               pruned_save_path="../saved/{}/Pruned.pth".format(exec_name),
                                               pruned_best_result_save_path="../saved/{}/pruned_best.pth".format(exec_name),
                                               sample_run=torch.zeros([1, 3, 224, 224]),
                                               pruning_params=pruning_params,
                                               exec_params=exec_params,
                                               dataset_params=dataset_params)
    return history

