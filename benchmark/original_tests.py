import os
import torch
import torchvision

from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from POC import ExecParams, PruningParams, DatasetParams
from Pruner.PartialPruning.ActivationMeanFilterPruner import ActivationMeanFilterPruner
from Pruner.PartialPruning.TaylorExpensionFilterPruner import TaylorExpensionFilterPruner
from benchmark.BenchmarkHelper import exec_squeeze_net, exec_dense_net, exec_resnet50, exec_vgg16, exec_resnet18, \
    exec_alexnet
from deeplib_ext.CustomDeepLib import train, test, display_sample_data
from FileHelper import save_obj
from deeplib_ext.MultiHistory import MultiHistory
from deeplib_ext.history import History

from thop_ext.profile import profile

def run_strategy_prune_compare(dataset_params):
    exec_param_no_prune_large = ExecParams(n_pretrain_epoch=0, n_epoch_retrain=0, n_epoch_total=15, batch_size=16,
                                           pruner=TaylorExpensionFilterPruner)
    exec_param_no_prune_medium = ExecParams(n_pretrain_epoch=0, n_epoch_retrain=0, n_epoch_total=15, batch_size=32,
                                            pruner=TaylorExpensionFilterPruner)
    exec_param_w_prune_large = ExecParams(n_pretrain_epoch=5, n_epoch_retrain=1, n_epoch_total=15, batch_size=16,
                                          pruner=TaylorExpensionFilterPruner)
    exec_param_w_prune_medium = ExecParams(n_pretrain_epoch=5, n_epoch_retrain=1, n_epoch_total=15, batch_size=32,
                                           pruner=TaylorExpensionFilterPruner)
    exec_param_no_prune = ExecParams(n_pretrain_epoch=0, n_epoch_retrain=0, n_epoch_total=15, batch_size=64,
                                     pruner=TaylorExpensionFilterPruner)
    exec_param_w_prune = ExecParams(n_pretrain_epoch=5, n_epoch_retrain=1, n_epoch_total=15, batch_size=64,
                                    pruner=TaylorExpensionFilterPruner)
    exec_param_w_prune_squeeze = ExecParams(n_pretrain_epoch=5, n_epoch_retrain=1, n_epoch_total=15, batch_size=64,
                                            pruner=TaylorExpensionFilterPruner, force_forward_view=True)
    pruning_param_no_prune = PruningParams(max_percent_per_iteration=0.0, prune_ratio=None)
    pruning_param_w_prune = PruningParams(max_percent_per_iteration=0.05, prune_ratio=0.3)

    multi_history = MultiHistory()
    exec_name = "SqueezeNet-0"
    h = exec_squeeze_net(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
                         dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    exec_name = "SqueezeNet-30"
    h = exec_squeeze_net(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_squeeze,
                         dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY,  title="Comparing Models at 30% Pruning")

    exec_name = "densenet 121-0"
    h = exec_dense_net(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune_large,
                       dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    exec_name = "densenet 121-30"
    h = exec_dense_net(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_large,
                       dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY,  title="Comparing Models at 30% Pruning")

    exec_name = "Resnet 50-0"
    h = exec_resnet50(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune_medium,
                      dataset_params=dataset_params, out_count=10)
    multi_history.append_history(exec_name, h)
    exec_name = "Resnet 50-30"
    h = exec_resnet50(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_medium,
                      dataset_params=dataset_params, out_count=10)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY,  title="Comparing Models at 30% Pruning")

    #create a second history since I am not sure it will look nice in one graph
    multi_history2 = MultiHistory()
    exec_name = "vgg16 0"
    h = exec_vgg16(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune_medium,
                   dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history2.append_history(exec_name, h)
    exec_name = "vgg16 30"
    h = exec_vgg16(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_medium,
                   dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history2.append_history(exec_name, h)

    exec_name = "Resnet 18-0"
    h = exec_resnet18(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
                      dataset_params=dataset_params, out_count=10)
    multi_history.append_history(exec_name, h)
    multi_history2.append_history(exec_name, h)
    exec_name = "Resnet 18-30"
    h = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune,
                      dataset_params=dataset_params, out_count=10)
    multi_history.append_history(exec_name, h)
    multi_history2.append_history(exec_name, h)

    exec_name = "Alexnet 0"
    h = exec_alexnet(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history2.append_history(exec_name, h)
    exec_name = "Alexnet 30"
    h = exec_alexnet(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history2.append_history(exec_name, h)

    save_obj(multi_history, "history_compare")
    multi_history.display_single_key(History.VAL_ACC_KEY,  title="Comparing Models at 30% Pruning")
    multi_history2.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")


def run_alex_prune_compare_rough(dataset_params):
    multi_history = MultiHistory()
    exec_param = ExecParams(n_pretrain_epoch=3, n_epoch_retrain=3, n_epoch_total=15, pruner=TaylorExpensionFilterPruner)
    exec_param_no_prune = ExecParams(n_pretrain_epoch=0, n_epoch_retrain=0, n_epoch_total=15,
                                     pruner=TaylorExpensionFilterPruner)

    exec_name = "Alexnet 0%"
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.0, prune_ratio=None),
                     exec_params=exec_param_no_prune, dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)

    exec_name = "Alexnet 10%"
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.05, prune_ratio=0.1), exec_params=exec_param,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)

    exec_name = "Alexnet 30%"
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.15, prune_ratio=0.3), exec_params=exec_param,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)

    exec_name = "Alexnet 50%"
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.25, prune_ratio=0.5), exec_params=exec_param,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)

    exec_name = "Alexnet 75%"
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.38, prune_ratio=0.75), exec_params=exec_param,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    save_obj(multi_history, "history_alex_rough")
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing AlexNet by Level of Pruning")


def run_alex_prune_compare_midway(dataset_params):
    multi_history = MultiHistory()
    exec_param = ExecParams(n_pretrain_epoch=3, n_epoch_retrain=2, n_epoch_total=15, pruner=TaylorExpensionFilterPruner)
    exec_param_no_prune = ExecParams(n_pretrain_epoch=0, n_epoch_retrain=0, n_epoch_total=15,
                                     pruner=TaylorExpensionFilterPruner)

    exec_name = "Alexnet 0%"
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.0, prune_ratio=None),
                     exec_params=exec_param_no_prune, dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)

    exec_name = "Alexnet 10%"
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.025, prune_ratio=0.1), exec_params=exec_param,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)

    exec_name = "Alexnet 30%"
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.075, prune_ratio=0.3), exec_params=exec_param,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)

    exec_name = "Alexnet 50%"
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.125, prune_ratio=0.5), exec_params=exec_param,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)

    exec_name = "Alexnet 75%"
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.1875, prune_ratio=0.75), exec_params=exec_param,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    save_obj(multi_history, "history_alex_midway")
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing AlexNet by Level of Pruning")


def run_alex_prune_compare_soft(dataset_params):
    multi_history = MultiHistory()
    exec_param = ExecParams(n_pretrain_epoch=3, n_epoch_retrain=1, n_epoch_total=15, pruner=TaylorExpensionFilterPruner)
    exec_param_no_prune = ExecParams(n_pretrain_epoch=0, n_epoch_retrain=0, n_epoch_total=15,
                                     pruner=TaylorExpensionFilterPruner)

    exec_name = "Alexnet 0%"
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.0, prune_ratio=None),
                     exec_params=exec_param_no_prune, dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)

    exec_name = "Alexnet 10%"
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.01, prune_ratio=0.1), exec_params=exec_param,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)

    exec_name = "Alexnet 30%"
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.03, prune_ratio=0.3), exec_params=exec_param,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)

    exec_name = "Alexnet 50%"
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.05, prune_ratio=0.5), exec_params=exec_param,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)

    exec_name = "Alexnet 75%"
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.075, prune_ratio=0.75), exec_params=exec_param,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    save_obj(multi_history, "history_alex_soft")
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing AlexNet by Level of Pruning")


def run_fast_validation(dataset_params):
    multi_history = MultiHistory()
    exec_param = ExecParams(n_pretrain_epoch=1, n_epoch_retrain=1, n_epoch_total=3, pruner=ActivationMeanFilterPruner)

    exec_name = "Alexnet test"
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.2, prune_ratio=0.2), exec_params=exec_param,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="TEST_RUN")


def run_compare_model_and_prune_alexnet():
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = CIFAR10("C:/dev/data/cifar10/", train=True, transform=transform, download=True)
    test_dataset = CIFAR10("C:/dev/data/cifar10/", train=False, transform=transform, download=True)
    dataset_params = DatasetParams(transform, train_dataset, test_dataset)

    run_strategy_prune_compare(dataset_params)
    run_alex_prune_compare_rough(dataset_params)
    run_alex_prune_compare_midway(dataset_params)
    run_alex_prune_compare_soft(dataset_params)


def run_test_using_image_net():
    exec_param_w_prune = ExecParams(n_pretrain_epoch=10,
                                    n_epoch_retrain=3,
                                    n_epoch_total=20,
                                    pruner=TaylorExpensionFilterPruner)
    pruning_param_w_prune = PruningParams(max_percent_per_iteration=0.2,
                                          prune_ratio=0.2)

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = torchvision.datasets.ImageFolder("TrainPATH", transform=transform)
    test_dataset = torchvision.datasets.ImageFolder("TestPATH", transform=transform)
    dataset_params = DatasetParams(transform, train_dataset, test_dataset)


    multi_history = MultiHistory()
    h = exec_resnet18(pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune,
                      dataset_params=dataset_params)
    multi_history.append_history("Resnet18 20%", h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Image_net")


def run_validation():
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = CIFAR10("C:/dev/data/cifar10/", train=True, transform=transform, download=True)
    test_dataset = CIFAR10("C:/dev/data/cifar10/", train=False, transform=transform, download=True)
    dataset_params = DatasetParams(transform, train_dataset, test_dataset)

    run_fast_validation(dataset_params)


if __name__ == '__main__':
    # run_validation()
    run_compare_model_and_prune_alexnet()
    # run_test_using_image_net()