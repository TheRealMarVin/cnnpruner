import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

import matplotlib.pyplot as plt

from POC import DatasetParams, PruningParams, ExecParams
from Pruner.CompletePruning.CompleteTaylorExpansionFilterPruner import CompleteTaylorExpansionFilterPruner
from benchmark.BenchmarkHelper import exec_squeeze_net, exec_alexnet, exec_resnet18, exec_resnet34, exec_resnet50, \
    exec_dense_net, exec_vgg16
from deeplib_ext.MultiHistory import MultiHistory
from deeplib_ext.history import History


def train_models(dataset_params):
    pruning_param_no_prune = PruningParams(max_percent_per_iteration=0.0, prune_ratio=None)
    exec_param_no_prune = ExecParams(n_pretrain_epoch=0, n_epoch_retrain=0, n_epoch_total=15, batch_size=32,
                                     pruner=None)

    multi_history = MultiHistory()

    exec_name = "AlexNet-base"
    h, s = exec_alexnet(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
                        dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Models training without pruning")

    exec_name = "ResNet18-base"
    h, s = exec_resnet18(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
                         dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Models training without pruning")

    exec_name = "ResNet34-base"
    h, s = exec_resnet34(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
                         dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Models training without pruning")

    exec_name = "ResNet50-base"
    h, s = exec_resnet50(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
                         dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Models training without pruning")

    exec_name = "Squeeze-base"
    h, s = exec_squeeze_net(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
                            dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Models training without pruning")

    exec_name = "DenseNet121-base"
    h, s = exec_dense_net(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
                          dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Models training without pruning")

    exec_name = "VGG16-base"
    h, s = exec_vgg16(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
                      dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Models training without pruning")


def run_compare_pruning(dataset_params):
    pruning_param_no_prune = PruningParams(max_percent_per_iteration=0.0, prune_ratio=0.0)
    exec_param = ExecParams(n_pretrain_epoch=0, n_epoch_retrain=2, n_epoch_total=2, batch_size=32,
                            pruner=CompleteTaylorExpansionFilterPruner)

    all_scores = {}
    exec_name = "AlexNet-degrad"
    score = []
    exec_param.best_result_save_path = "../saved/AlexNet-base/Pruned.pth".format(exec_name)
    exec_param.retrain_if_weight_loaded = True
    for i in range(0, 11):
        desired_pruning = (5.0 * i)/100.0
        if desired_pruning != 0.0:
            pruning_param_no_prune.max_percent_per_iteration = desired_pruning
            pruning_param_no_prune.prune_ratio = desired_pruning
        else:
            pruning_param_no_prune.max_percent_per_iteration = None
            pruning_param_no_prune.prune_ratio = None
        h, s = exec_alexnet(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param,
                             dataset_params=dataset_params)
        score.append(s)

    all_scores["AlexNet"] = score
    display_graphs(score, all_scores, "AlexNet")


    exec_name = "ResNet18-degrad"
    score = []
    exec_param.best_result_save_path = "../saved/ResNet18-base/Pruned.pth".format(exec_name)
    exec_param.retrain_if_weight_loaded = True
    for i in range(0, 11):
        desired_pruning = (5.0 * i)/100.0
        if desired_pruning != 0.0:
            pruning_param_no_prune.max_percent_per_iteration = desired_pruning
            pruning_param_no_prune.prune_ratio = desired_pruning
        else:
            pruning_param_no_prune.max_percent_per_iteration = None
            pruning_param_no_prune.prune_ratio = None
        h, s = exec_resnet18(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param,
                             dataset_params=dataset_params)
        score.append(s)

    all_scores["ResNet18"] = score
    display_graphs(score, all_scores, "ResNet18")

    exec_name = "ResNet34-degrad"
    score = []
    exec_param.best_result_save_path = "../saved/ResNet34-base/Pruned.pth".format(exec_name)
    exec_param.retrain_if_weight_loaded = True
    for i in range(0, 11):
        desired_pruning = (5.0 * i)/100.0
        if desired_pruning != 0.0:
            pruning_param_no_prune.max_percent_per_iteration = desired_pruning
            pruning_param_no_prune.prune_ratio = desired_pruning
        else:
            pruning_param_no_prune.max_percent_per_iteration = None
            pruning_param_no_prune.prune_ratio = None
        h, s = exec_resnet34(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param,
                             dataset_params=dataset_params)
        score.append(s)

    all_scores["ResNet34"] = score
    display_graphs(score, all_scores, "ResNet34")

    exec_name = "ResNet50-degrad"
    score = []
    exec_param.best_result_save_path = "../saved/ResNet50-base/Pruned.pth".format(exec_name)
    exec_param.retrain_if_weight_loaded = True
    for i in range(0, 11):
        desired_pruning = (5.0 * i)/100.0
        if desired_pruning != 0.0:
            pruning_param_no_prune.max_percent_per_iteration = desired_pruning
            pruning_param_no_prune.prune_ratio = desired_pruning
        else:
            pruning_param_no_prune.max_percent_per_iteration = None
            pruning_param_no_prune.prune_ratio = None
        h, s = exec_resnet50(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param,
                             dataset_params=dataset_params)
        score.append(s)

    all_scores["ResNet50"] = score
    display_graphs(score, all_scores, "ResNet50")

    exec_name = "DenseNet121-degrad"
    score = []
    exec_param.best_result_save_path = "../saved/DenseNet121-base/Pruned.pth".format(exec_name)
    exec_param.retrain_if_weight_loaded = True
    for i in range(0, 11):
        desired_pruning = (5.0 * i)/100.0
        if desired_pruning != 0.0:
            pruning_param_no_prune.max_percent_per_iteration = desired_pruning
            pruning_param_no_prune.prune_ratio = desired_pruning
        else:
            pruning_param_no_prune.max_percent_per_iteration = None
            pruning_param_no_prune.prune_ratio = None
        h, s = exec_dense_net(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param,
                              dataset_params=dataset_params)
        score.append(s)

    all_scores["DenseNet121"] = score
    display_graphs(score, all_scores, "DenseNet121")

    exec_name = "VGG16-degrad"
    score = []
    exec_param.best_result_save_path = "../saved/VGG16-base/Pruned.pth".format(exec_name)
    exec_param.retrain_if_weight_loaded = True
    for i in range(0, 11):
        desired_pruning = (5.0 * i)/100.0
        if desired_pruning != 0.0:
            pruning_param_no_prune.max_percent_per_iteration = desired_pruning
            pruning_param_no_prune.prune_ratio = desired_pruning
        else:
            pruning_param_no_prune.max_percent_per_iteration = None
            pruning_param_no_prune.prune_ratio = None
        h, s = exec_dense_net(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param,
                              dataset_params=dataset_params)
        score.append(s)

    all_scores["VGG16"] = score
    display_graphs(score, all_scores, "VGG16")

    exec_name = "Squeeze-degrad"
    score = []
    exec_param.best_result_save_path = "../saved/Squeeze-base/Pruned.pth".format(exec_name)
    exec_param.retrain_if_weight_loaded = True
    for i in range(0, 11):
        desired_pruning = (5.0 * i) / 100.0
        if desired_pruning != 0.0:
            pruning_param_no_prune.max_percent_per_iteration = desired_pruning
            pruning_param_no_prune.prune_ratio = desired_pruning
        else:
            pruning_param_no_prune.max_percent_per_iteration = None
            pruning_param_no_prune.prune_ratio = None
        h, s = exec_vgg16(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param,
                          dataset_params=dataset_params)
        score.append(s)

    all_scores["Squeeze"] = score
    display_graphs(score, all_scores, "Squeeze")


def display_graphs(current_score, all_scores, name):
    plt.title("Display degradatrion of {}".format(name))
    plt.xlabel("Pruning Percent")
    plt.ylabel("Test Score")
    plt.plot(current_score)
    plt.show()

    plt.title("Display full degradatrion")
    plt.xlabel("Pruning Percent")
    plt.ylabel("Test Score")
    for k, v in all_scores.items():
        pruning = [x * 0.05 for x in range(0, len(v))]
        plt.plot(pruning, v, label=k)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = CIFAR10("C:/dev/data/cifar10/", train=True, transform=transform, download=True)
    test_dataset = CIFAR10("C:/dev/data/cifar10/", train=False, transform=transform, download=True)
    dataset_params = DatasetParams(transform, train_dataset, test_dataset)

    # train_models(dataset_params)
    run_compare_pruning(dataset_params)
