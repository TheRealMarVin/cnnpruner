import os
import torch
import torchvision

from torch import nn
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from Pruner.ActivationMeanFilterPruner import ActivationMeanFilterPruner
from Pruner.TaylorExpensionFilterPruner import TaylorExpensionFilterPruner
from deeplib_ext.CustomDeepLib import train, test, display_sample_data
from FileHelper import load_obj, save_obj
# from ModelHelper import total_num_filters
from deeplib_ext.MultiHistory import MultiHistory
from deeplib_ext.history import History
from models.AlexNetSki import alexnetski

from thop_ext.profile import profile

# TODO check this one!!! https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
# and this: https://github.com/fg91/visualizing-cnn-feature-maps/blob/master/Calculate_mean_activation_per_filter_in_specific_layer_given_an_image.ipynb


class ExecParams:
    def __init__(self,
                 n_pretrain_epoch=10,
                 n_epoch_retrain=3,
                 n_epoch_total=20,
                 batch_size=64,
                 learning_rate=0.01,
                 pruner=TaylorExpensionFilterPruner,
                 force_forward_view=False):
        self.n_pretrain_epoch = n_pretrain_epoch
        self.n_epoch_retrain = n_epoch_retrain
        self.n_epoch_total = n_epoch_total
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pruner = pruner
        self.force_forward_view = force_forward_view


class PruningParams:
    def __init__(self, max_percent_per_iteration=0.1, prune_ratio=0.3):
        self.max_percent_per_iteration = max_percent_per_iteration
        self.prune_ratio = prune_ratio


class DatasetParams:
    def __init__(self, transform, train_dataset, test_dataset):
        self.transform = transform
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def display_dataset_sample(self):
        display_sample_data(self.train_dataset)
        display_sample_data(self.test_dataset)


class DebugHelper:
    def __init__(self, transform, train_dataset, test_dataset):
        pass

#TODO some needs to be cleaned or are no longer used
def common_training_code(model,
                         pruned_save_path=None,
                         best_result_save_path=None,
                         pruned_best_result_save_path=None,
                         retrain_if_weight_loaded=False,
                         sample_run=None,
                         # reuse_cut_filter=False,
                         pruning_params=None,
                         exec_params=None,
                         dataset_params=None):
    model.cuda()

    flops, params = profile(model, input_size=(1, 3, 224, 224))
    print("number of flops: {} \tnumber of params: {}".format(flops, params))

    use_gpu = True
    n_epoch_total = exec_params.n_epoch_total
    history = History()

    optimizer = torch.optim.SGD(model.parameters(), exec_params.learning_rate)
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
    if should_train and exec_params.n_pretrain_epoch > 0:
        local_history = train(model, optimizer, dataset_params.train_dataset, exec_params.n_pretrain_epoch,
                              exec_params.batch_size, use_gpu=use_gpu, criterion=criterion,
                              scheduler=scheduler, best_result_save_path=best_result_save_path)
        history.append(local_history)
        n_epoch_total = n_epoch_total - exec_params.n_pretrain_epoch

    test_score = test(model, dataset_params.test_dataset, exec_params.batch_size, use_gpu=use_gpu)
    print('Test:\n\tScore: {}'.format(test_score))

    ###
    #TODO maybe put the loop content in a function that looks terrible now
    if pruning_params.prune_ratio is not None:
        pruner = exec_params.pruner(model, sample_run, exec_params.force_forward_view)

        # prune_targets = None
        # if reuse_cut_filter:
        #     prune_targets = load_obj("filters_dic")

        train(model, optimizer, dataset_params.train_dataset, 1, exec_params.batch_size, use_gpu=use_gpu,
              criterion=criterion, scheduler=scheduler, pruner=pruner, batch_count=1, should_validate=False)

        number_of_filters = pruner.get_number_of_filter_to_prune()
        initial_number_of_filters = number_of_filters
        filter_to_prune = int(number_of_filters * pruning_params.prune_ratio)
        max_filters_to_prune_on_iteration = int(number_of_filters * pruning_params.max_percent_per_iteration)
        if filter_to_prune < max_filters_to_prune_on_iteration:
            max_filters_to_prune_on_iteration = filter_to_prune
        iterations = (filter_to_prune//max_filters_to_prune_on_iteration)
        print("{} iterations to reduce {:2.2f}% filters".format(iterations, pruning_params.prune_ratio*100))

        for param in model.parameters():
            param.requires_grad = True

        for iteration_idx in range(iterations):
            print("Perform pruning iteration: {}".format(iteration_idx))

            pruner.normalize_layer()
            prune_targets = pruner.plan_prunning(max_filters_to_prune_on_iteration)
            # pruner.reset()

            # prune_targets = None
            # if reuse_cut_filter:
            #     prune_targets = load_obj("filters_dic")

            # if prune_targets is None:
            #     # train(model, optimizer, dataset_params.train_dataset, 1, exec_params.batch_size, use_gpu=use_gpu, criterion=criterion,
            #     #       scheduler=scheduler, pruner=pruner, batch_count=1, should_validate=False)
            #
            #     pruner.normalize_layer()
            #
            #     prune_targets = pruner.plan_prunning(max_filters_to_prune_on_iteration)
            #     if reuse_cut_filter:
            #         save_obj(prune_targets, "filters_dic")

            pruner.display_pruning_log(prune_targets)

            print("Pruning filters.. ")
            model = model.cpu()
            pruner.prune(prune_targets)

            model = model.cuda()
            # optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.007)
            optimizer = torch.optim.SGD(model.parameters(), exec_params.learning_rate)
            pruner.reset()

            print("Filters pruned {}%".format(100 * float(max_filters_to_prune_on_iteration) / initial_number_of_filters))
            new_test_score = test(model, dataset_params.test_dataset, exec_params.batch_size, use_gpu=use_gpu)
            print('Test:\n\tpost prune Score: {}'.format(new_test_score))

            # basedir = os.path.dirname(pruned_save_path)
            # if not os.path.exists(basedir):
            #     os.makedirs(basedir)
            # torch.save(model, pruned_save_path)

            print("Fine tuning to recover from prunning iteration.")
            local_history = train(model, optimizer, dataset_params.train_dataset,
                                  exec_params.n_epoch_retrain, exec_params.batch_size, use_gpu=use_gpu,
                                  criterion=None, scheduler=scheduler, pruner=None,
                                  best_result_save_path=pruned_best_result_save_path)
            n_epoch_total = n_epoch_total - exec_params.n_epoch_retrain
            history.append(local_history)
            # local_history.display()
            test_score = test(model, dataset_params.test_dataset, exec_params.batch_size, use_gpu=use_gpu)
            print('Test pruning iteration :{}\n\tScore: {}'.format(iteration_idx, test_score))

            if iteration_idx < iterations - 1:
                pruner.reset()
                train(model, optimizer, dataset_params.train_dataset, 1, exec_params.batch_size, use_gpu=use_gpu,
                      criterion=criterion, scheduler=scheduler, pruner=pruner, batch_count=1, should_validate=False)
    ###

    if n_epoch_total > 0:
        local_history = train(model, optimizer, dataset_params.train_dataset, n_epoch_total,
                              exec_params.batch_size, use_gpu=use_gpu, criterion=criterion,
                              scheduler=scheduler, best_result_save_path=pruned_best_result_save_path)
        history.append(local_history)

    basedir = os.path.dirname(pruned_save_path)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    torch.save(model, pruned_save_path)

    history.display()
    flops, params = profile(model, input_size=(1, 3, 224, 224))
    print("end number of flops: {} \tnumber of params: {}".format(flops, params))

    test_score = test(model, dataset_params.test_dataset, exec_params.batch_size, use_gpu=use_gpu)
    print('Final Test:\n\tScore: {}'.format(test_score))

    return history


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


def exec_squeeze_net(exec_name, pruning_params=None, exec_params=None, dataset_params=None):
    print("*** ", exec_name)

    model = models.squeezenet1_1(pretrained=True)
    model.cuda()

    history = common_training_code(model, pruned_save_path="saved/{}/Pruned.pth".format(exec_name),
                                   pruned_best_result_save_path="saved/{}/pruned_best.pth".format(exec_name),
                                   sample_run=torch.zeros([1, 3, 224, 224]),
                                   pruning_params=pruning_params,
                                   exec_params=exec_params,
                                   dataset_params=dataset_params)
    return history


def exec_dense_net(exec_name, pruning_params=None, exec_params=None, dataset_params=None):
    print("*** ", exec_name)

    model = models.densenet121(pretrained=True)
    model.cuda()

    history = common_training_code(model, pruned_save_path="saved/{}/Pruned.pth".format(exec_name),
                                   pruned_best_result_save_path="saved/{}/pruned_best.pth".format(exec_name),
                                   sample_run=torch.zeros([1, 3, 224, 224]),
                                   pruning_params=pruning_params,
                                   exec_params=exec_params,
                                   dataset_params=dataset_params)
    return history


def exec_vgg16(exec_name, pruning_params=None, exec_params=None, dataset_params=None):
    print("*** ", exec_name)

    model = models.vgg16(pretrained=True)
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


def exec_resnet50(exec_name, pruning_params=None, exec_params=None, dataset_params=None, out_count=1000):
    print("*** ", exec_name)

    model = models.resnet50(pretrained=True)
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
    multi_history.display_single_key(History.VAL_ACC_KEY,  title="Comparing Models ar 30% Pruning")

    exec_name = "densenet 121-0"
    h = exec_dense_net(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune_large,
                       dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    exec_name = "densenet 121-30"
    h = exec_dense_net(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_large,
                       dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY,  title="Comparing Models ar 30% Pruning")

    exec_name = "Resnet 50-0"
    h = exec_resnet50(exec_name, pruning_params=pruning_param_no_prune, exec_params=exec_param_no_prune_medium,
                      dataset_params=dataset_params, out_count=10)
    multi_history.append_history(exec_name, h)
    exec_name = "Resnet 50-30"
    h = exec_resnet50(exec_name, pruning_params=pruning_param_w_prune, exec_params=exec_param_w_prune_medium,
                      dataset_params=dataset_params, out_count=10)
    multi_history.append_history(exec_name, h)
    multi_history.display_single_key(History.VAL_ACC_KEY,  title="Comparing Models ar 30% Pruning")

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
    multi_history.display_single_key(History.VAL_ACC_KEY,  title="Comparing Models ar 30% Pruning")
    multi_history2.display_single_key(History.VAL_ACC_KEY, title="Comparing Models ar 30% Pruning")


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
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.02, prune_ratio=0.1), exec_params=exec_param,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)

    exec_name = "Alexnet 30%"
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.06, prune_ratio=0.3), exec_params=exec_param,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)

    exec_name = "Alexnet 50%"
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.1, prune_ratio=0.5), exec_params=exec_param,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)

    exec_name = "Alexnet 75%"
    h = exec_alexnet(exec_name, PruningParams(max_percent_per_iteration=0.15, prune_ratio=0.75), exec_params=exec_param,
                     dataset_params=dataset_params)
    multi_history.append_history(exec_name, h)
    save_obj(multi_history, "history_alex_soft")
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
