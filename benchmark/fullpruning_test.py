import torch
import torchvision

from torch import nn
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from POC import common_training_code, ExecParams, PruningParams, DatasetParams
from Pruner.PartialPruning.ActivationMeanFilterPruner import ActivationMeanFilterPruner
from Pruner.CompletePruning.Alt.CompleteActivationMeanFilterPrunerV2 import ActivationMeanFilterPrunerV2
from Pruner.CompletePruning.CompleteActivationMeanFilterPruner import ActivationMeanFilterPrunerV3
from Pruner.CompletePruning.Alt.CompleteActivationMeanFilterPrunerV4 import ActivationMeanFilterPrunerV4
from Pruner.PartialPruning.TaylorExpensionFilterPruner import TaylorExpensionFilterPruner
from Pruner.CompletePruning.Alt.CompleteTaylorExpensionFilterPrunerV2 import TaylorExpensionFilterPrunerv2
from Pruner.CompletePruning.CompleteTaylorExpensionFilterPruner import CompleteTaylorExpensionFilterPruner
from Pruner.CompletePruning.Alt.CompleteTaylorExpensionFilterPrunerV4 import TaylorExpensionFilterPrunerv4
from FileHelper import save_obj
# from ModelHelper import total_num_filters
from deeplib_ext.MultiHistory import MultiHistory
from deeplib_ext.history import History
from models.AlexNetSki import alexnetski


# TODO check this one!!! https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
# and this: https://github.com/fg91/visualizing-cnn-feature-maps/blob/master/Calculate_mean_activation_per_filter_in_specific_layer_given_an_image.ipynb




def exec_alexnet(exec_name, pruning_params=None, exec_params=None, dataset_params=None):
    print("*** ", exec_name)
    model = alexnetski(pretrained=True)
    model.cuda()

    history = common_training_code(model, pruned_save_path="saved/{}/Pruned.pth".format(exec_name),
                                   pruned_best_result_save_path="saved/{}/pruned_best.pth".format(exec_name),
                                   sample_run=torch.zeros([1, 3, 224, 224]),
                                   pruning_params=pruning_params,
                                   exec_params=exec_params,
                                   dataset_params=dataset_params)

    return history




def exec_resnet18(exec_name, pruning_params=None, exec_params=None, dataset_params=None, out_count=1000):
    print("*** ", exec_name)

    model = models.resnet18(pretrained=True)
    if model.fc.out_features != out_count:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)

    model.cuda()

    history = common_training_code(model, pruned_save_path="saved/{}/Pruned.pth".format(exec_name),
                                   pruned_best_result_save_path="saved/{}/pruned_best.pth".format(exec_name),
                                   sample_run=torch.zeros([1, 3, 224, 224]),
                                   pruning_params=pruning_params,
                                   exec_params=exec_params,
                                   dataset_params=dataset_params)
    return history


def exec_resnet34(exec_name, pruning_params=None, exec_params=None, dataset_params=None, out_count=1000):
    print("*** ", exec_name)

    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    model.cuda()

    history = common_training_code(model, pruned_save_path="saved/{}/Pruned.pth".format(exec_name),
                                   pruned_best_result_save_path="saved/{}/pruned_best.pth".format(exec_name),
                                   sample_run=torch.zeros([1, 3, 224, 224]),
                                   pruning_params=pruning_params,
                                   exec_params=exec_params,
                                   dataset_params=dataset_params)
    return history


def exec_squeeze_net(exec_name, pruning_params=None, exec_params=None, dataset_params=None):
    print("*** ", exec_name)

    model = models.squeezenet1_1(pretrained=True)
    model.cuda()

    if exec_params is not None:
        exec_params.force_forward_view = True
        exec_params.ignore_last_conv = True

    history = common_training_code(model, pruned_save_path="saved/{}/Pruned.pth".format(exec_name),
                                   pruned_best_result_save_path="saved/{}/pruned_best.pth".format(exec_name),
                                   sample_run=torch.zeros([1, 3, 224, 224]),
                                   pruning_params=pruning_params,
                                   exec_params=exec_params,
                                   dataset_params=dataset_params)
    return history

def run_strategy_prune_compare_taylor(dataset_params):
    exec_param_no_prune = ExecParams(n_pretrain_epoch=0, n_epoch_retrain=0, n_epoch_total=15, batch_size=64,
                                     pruner=ActivationMeanFilterPrunerV2)
    exec_param_w_prune_2 = ExecParams(n_pretrain_epoch=5, n_epoch_retrain=1, n_epoch_total=15, batch_size=64,
                                      pruner=TaylorExpensionFilterPrunerv2)
    exec_param_w_prune_3 = ExecParams(n_pretrain_epoch=5, n_epoch_retrain=1, n_epoch_total=15, batch_size=64,
                                      pruner=CompleteTaylorExpensionFilterPruner)
    exec_param_w_prune_4 = ExecParams(n_pretrain_epoch=5, n_epoch_retrain=1, n_epoch_total=15, batch_size=64,
                                      pruner=CompleteTaylorExpensionFilterPruner)
    exec_param_w_prune_t = ExecParams(n_pretrain_epoch=5, n_epoch_retrain=1, n_epoch_total=15, batch_size=64,
                                      pruner=TaylorExpensionFilterPruner)
    exec_param_w_prune_o = ExecParams(n_pretrain_epoch=5, n_epoch_retrain=1, n_epoch_total=15, batch_size=64,
                                      pruner=ActivationMeanFilterPruner)
    pruning_param_no_prune = PruningParams(max_percent_per_iteration=0.0, prune_ratio=None)
    pruning_param_w_prune = PruningParams(max_percent_per_iteration=0.075, prune_ratio=0.30)
    pruning_param_w_prune2 = PruningParams(max_percent_per_iteration=0.04, prune_ratio=0.17)

    multi_history = MultiHistory()

    exec_name = "Squeeze - 30 full"
    h = exec_squeeze_net(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_4,
                         dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    exec_name = "Squeeze - 30 Simple"
    h = exec_squeeze_net(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_o,
                         dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")


    # exec_name = "Resnet 18-0"
    # h = exec_resnet18(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # exec_name = "Resnet 18-30-Simple_prune"
    # h = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_o,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    # exec_name = "Resnet 18-30-v2"
    # h = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_2,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    # exec_name = "Resnet 18-30-v3"
    # h = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_3,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    # exec_name = "Resnet 18-30-v3-2"
    # h = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune2, exec_params=exec_param_w_prune_3,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    # exec_name = "Resnet 18-30-v4-p1"
    # h = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_4,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    # exec_name = "Resnet 18-30-v4-p2"
    # h = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune2, exec_params=exec_param_w_prune_4,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    # exec_name = "Resnet 18-30-Taylor"
    # h = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_t,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)

    # exec_name = "Alexnet 0"
    # h = exec_alexnet(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
    #                  dataset_params=dataset_params)
    # multi_history.append_history(exec_name, h)
    # exec_name = "Alexnet 30"
    # h = exec_alexnet(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune,
    #                  dataset_params=dataset_params)
    # multi_history.append_history(exec_name, h)

    save_obj(multi_history, "history_compare")
    multi_history.display_single_key(History.VAL_ACC_KEY,  title="Comparing Models at 30% Pruning")

def run_strategy_prune_compare_activation_mean(dataset_params):
    exec_param_no_prune = ExecParams(n_pretrain_epoch=0, n_epoch_retrain=0, n_epoch_total=15, batch_size=64,
                                     pruner=ActivationMeanFilterPrunerV2)
    exec_param_w_prune_2 = ExecParams(n_pretrain_epoch=5, n_epoch_retrain=1, n_epoch_total=15, batch_size=64,
                                      pruner=ActivationMeanFilterPrunerV2)
    exec_param_w_prune_3 = ExecParams(n_pretrain_epoch=5, n_epoch_retrain=1, n_epoch_total=15, batch_size=64,
                                      pruner=ActivationMeanFilterPrunerV3)
    exec_param_w_prune_4 = ExecParams(n_pretrain_epoch=5, n_epoch_retrain=1, n_epoch_total=15, batch_size=64,
                                      pruner=ActivationMeanFilterPrunerV4)
    exec_param_w_prune_t = ExecParams(n_pretrain_epoch=5, n_epoch_retrain=1, n_epoch_total=15, batch_size=64,
                                      pruner=CompleteTaylorExpensionFilterPruner)
    exec_param_w_prune_o = ExecParams(n_pretrain_epoch=5, n_epoch_retrain=1, n_epoch_total=15, batch_size=64,
                                      pruner=ActivationMeanFilterPruner)
    pruning_param_no_prune = PruningParams(max_percent_per_iteration=0.0, prune_ratio=None)
    pruning_param_w_prune = PruningParams(max_percent_per_iteration=0.05, prune_ratio=0.15)

    multi_history = MultiHistory()

    # exec_name = "Resnet 18-0"
    # h = exec_resnet18(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # exec_name = "Resnet 18-30-Simple_prune"
    # h = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_o,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    # exec_name = "Resnet 18-30-v2"
    # h = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_2,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    exec_name = "Resnet 18-30-v3"
    h = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_3,
                      dataset_params=dataset_params, out_count=10)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    # exec_name = "Resnet 18-30-v4"
    # h = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_4,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    # exec_name = "Resnet 18-30-Taylor"
    # h = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_t,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)

    # exec_name = "Alexnet 0"
    # h = exec_alexnet(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
    #                  dataset_params=dataset_params)
    # multi_history.append_history(exec_name, h)
    # exec_name = "Alexnet 30"
    # h = exec_alexnet(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune,
    #                  dataset_params=dataset_params)
    # multi_history.append_history(exec_name, h)

    save_obj(multi_history, "history_compare")
    multi_history.display_single_key(History.VAL_ACC_KEY,  title="Comparing Models at 30% Pruning")


def run_fast_validation(dataset_params):
    multi_history = MultiHistory()
    exec_param = ExecParams(n_pretrain_epoch=1, n_epoch_retrain=1, n_epoch_total=3, pruner=ActivationMeanFilterPruner)

    exec_name = "Alexnet test"
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.2, prune_ratio=0.2), exec_params=exec_param,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="TEST_RUN")


def run_compare_pruning():
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = CIFAR10("C:/dev/data/cifar10/", train=True, transform=transform, download=True)
    test_dataset = CIFAR10("C:/dev/data/cifar10/", train=False, transform=transform, download=True)
    dataset_params = DatasetParams(transform, train_dataset, test_dataset)

    run_strategy_prune_compare_taylor(dataset_params)
    # run_strategy_prune_compare_activation_mean(dataset_params)


def run_test_using_image_net():
    exec_param_w_prune = ExecParams(n_pretrain_epoch=10,
                                    n_epoch_retrain=3,
                                    n_epoch_total=20,
                                    pruner=CompleteTaylorExpensionFilterPruner)
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
    run_compare_pruning()
    # run_test_using_image_net()
