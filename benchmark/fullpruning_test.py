import torchvision

from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from POC import common_training_code, ExecParams, PruningParams, DatasetParams
from Pruner.PartialPruning.ActivationMeanFilterPruner import ActivationMeanFilterPruner
from Pruner.CompletePruning.Alt.CompleteActivationMeanFilterPrunerV2 import ActivationMeanFilterPrunerV2
from Pruner.CompletePruning.CompleteActivationMeanFilterPruner import ActivationMeanFilterPrunerV3
from Pruner.CompletePruning.Alt.CompleteActivationMeanFilterPrunerV4 import ActivationMeanFilterPrunerV4
from Pruner.PartialPruning.TaylorExpansionFilterPruner import TaylorExpansionFilterPruner
from Pruner.CompletePruning.Alt.CompleteTaylorExpensionFilterPrunerV2 import TaylorExpansionFilterPrunerV2
from Pruner.CompletePruning.CompleteTaylorExpansionFilterPruner import CompleteTaylorExpansionFilterPruner
from FileHelper import save_obj

from benchmark.BenchmarkHelper import exec_squeeze_net, exec_resnet18, exec_alexnet
from deeplib_ext.MultiHistory import MultiHistory
from deeplib_ext.history import History


# TODO check this one!!! https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
# and this: https://github.com/fg91/visualizing-cnn-feature-maps/blob/master/Calculate_mean_activation_per_filter_in_specific_layer_given_an_image.ipynb
def run_strategy_prune_compare_taylor(dataset_params):
    exec_param_no_prune = ExecParams(n_pretrain_epoch=0, n_epoch_retrain=0, n_epoch_total=15, batch_size=64,
                                     pruner=ActivationMeanFilterPrunerV2)
    exec_param_w_prune_2 = ExecParams(n_pretrain_epoch=5, n_epoch_retrain=1, n_epoch_total=15, batch_size=64,
                                      pruner=TaylorExpansionFilterPrunerV2)
    exec_param_w_prune_3 = ExecParams(n_pretrain_epoch=5, n_epoch_retrain=1, n_epoch_total=15, batch_size=64,
                                      pruner=CompleteTaylorExpansionFilterPruner)
    exec_param_w_prune_4 = ExecParams(n_pretrain_epoch=5, n_epoch_retrain=1, n_epoch_total=15, batch_size=64,
                                      pruner=CompleteTaylorExpansionFilterPruner)
    exec_param_w_prune_t = ExecParams(n_pretrain_epoch=5, n_epoch_retrain=1, n_epoch_total=15, batch_size=64,
                                      pruner=TaylorExpansionFilterPruner)
    exec_param_w_prune_o = ExecParams(n_pretrain_epoch=5, n_epoch_retrain=1, n_epoch_total=15, batch_size=64,
                                      pruner=ActivationMeanFilterPruner)
    pruning_param_no_prune = PruningParams(max_percent_per_iteration=0.0, prune_ratio=None)
    pruning_param_w_prune = PruningParams(max_percent_per_iteration=0.075, prune_ratio=0.30)
    pruning_param_w_prune2 = PruningParams(max_percent_per_iteration=0.05, prune_ratio=0.15)

    multi_history = MultiHistory()

    exec_name = "Squeeze - 30 full"
    h, s = exec_squeeze_net(exec_name, pruning_params=pruning_param_w_prune2, exec_params=exec_param_w_prune_4,
                            dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    exec_name = "Squeeze - 30 Simple"
    h, s = exec_squeeze_net(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_o,
                            dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")


    # exec_name = "Resnet 18-0"
    # h, s = exec_resnet18(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # exec_name = "Resnet 18-30-Simple_prune"
    # h, s = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_o,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    # exec_name = "Resnet 18-30-v2"
    # h, s = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_2,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    # exec_name = "Resnet 18-30-v3"
    # h, s = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_3,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    # exec_name = "Resnet 18-30-v3-2"
    # h, s = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune2, exec_params=exec_param_w_prune_3,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    # exec_name = "Resnet 18-30-v4-p1"
    # h, s = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_4,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    # exec_name = "Resnet 18-30-v4-p2"
    # h, s = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune2, exec_params=exec_param_w_prune_4,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    # exec_name = "Resnet 18-30-Taylor"
    # h, s = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_t,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)

    # exec_name = "Alexnet 0"
    # h, s = exec_alexnet(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
    #                  dataset_params=dataset_params)
    # multi_history.append_history(exec_name, h)
    # exec_name = "Alexnet 30"
    # h, s = exec_alexnet(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune,
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
                                      pruner=CompleteTaylorExpansionFilterPruner)
    exec_param_w_prune_o = ExecParams(n_pretrain_epoch=5, n_epoch_retrain=1, n_epoch_total=15, batch_size=64,
                                      pruner=ActivationMeanFilterPruner)
    pruning_param_no_prune = PruningParams(max_percent_per_iteration=0.0, prune_ratio=None)
    pruning_param_w_prune = PruningParams(max_percent_per_iteration=0.05, prune_ratio=0.15)

    multi_history = MultiHistory()

    # exec_name = "Resnet 18-0"
    # h, s = exec_resnet18(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # exec_name = "Resnet 18-30-Simple_prune"
    # h, s = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_o,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    # exec_name = "Resnet 18-30-v2"
    # h, s = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_2,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    exec_name = "Resnet 18-30-v3"
    h, s = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_3,
                      dataset_params=dataset_params, out_count=10)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    # exec_name = "Resnet 18-30-v4"
    # h, s = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_4,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)
    # multi_history.display_single_key(History.VAL_ACC_KEY, title="Comparing Models at 30% Pruning")
    # exec_name = "Resnet 18-30-Taylor"
    # h, s = exec_resnet18(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_t,
    #                   dataset_params=dataset_params, out_count=10)
    # multi_history.append_history(exec_name, h)

    # exec_name = "Alexnet 0"
    # h, s = exec_alexnet(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
    #                  dataset_params=dataset_params)
    # multi_history.append_history(exec_name, h)
    # exec_name = "Alexnet 30"
    # h, s = exec_alexnet(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune,
    #                  dataset_params=dataset_params)
    # multi_history.append_history(exec_name, h)

    save_obj(multi_history, "history_compare")
    multi_history.display_single_key(History.VAL_ACC_KEY,  title="Comparing Models at 30% Pruning")


def run_fast_validation(dataset_params):
    multi_history = MultiHistory()
    exec_param = ExecParams(n_pretrain_epoch=1, n_epoch_retrain=1, n_epoch_total=3, pruner=ActivationMeanFilterPruner)

    exec_name = "Alexnet test"
    h, s = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.2, prune_ratio=0.2), exec_params=exec_param,
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
                                    pruner=CompleteTaylorExpansionFilterPruner)
    pruning_param_w_prune = PruningParams(max_percent_per_iteration=0.2,
                                          prune_ratio=0.2)

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = torchvision.datasets.ImageFolder("TrainPATH", transform=transform)
    test_dataset = torchvision.datasets.ImageFolder("TestPATH", transform=transform)
    dataset_params = DatasetParams(transform, train_dataset, test_dataset)


    multi_history = MultiHistory()
    h, s = exec_resnet18(pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune,
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
