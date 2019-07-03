from torch import nn
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from POC import DatasetParams, PruningParams, ExecParams
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
    h = exec_alexnet(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
                         dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Models training without pruning")

    exec_name = "ResNet18-base"
    h = exec_resnet18(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Models training without pruning")

    exec_name = "ResNet34-base"
    h = exec_resnet34(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
                      dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Models training without pruning")

    exec_name = "ResNet50-base"
    h = exec_resnet50(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
                      dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Models training without pruning")

    exec_name = "Squeeze-base"
    h = exec_squeeze_net(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
                         dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Models training without pruning")

    exec_name = "DenseNet121-base"
    h = exec_dense_net(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
                         dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Models training without pruning")

    exec_name = "VGG16-base"
    h = exec_vgg16(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune,
                       dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY, title="Models training without pruning")


def run_compare_pruning(dataset_params):
    pass


if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = CIFAR10("C:/dev/data/cifar10/", train=True, transform=transform, download=True)
    test_dataset = CIFAR10("C:/dev/data/cifar10/", train=False, transform=transform, download=True)
    dataset_params = DatasetParams(transform, train_dataset, test_dataset)

    train_models(dataset_params)
    run_compare_pruning(dataset_params)
    # run_test_using_image_net()