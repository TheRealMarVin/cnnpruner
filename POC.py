import os
import torch
import torchvision

from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from Pruner.PartialPruning.ActivationMeanFilterPruner import ActivationMeanFilterPruner
from Pruner.PartialPruning.TaylorExpansionFilterPruner import TaylorExpansionFilterPruner
from deeplib_ext.CustomDeepLib import train, test, display_sample_data
from FileHelper import save_obj
from deeplib_ext.MultiHistory import MultiHistory
from deeplib_ext.history import History

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
                 pruner=TaylorExpansionFilterPruner,
                 force_forward_view=False,
                 ignore_last_conv=False,
                 best_result_save_path=None,
                 retrain_if_weight_loaded=False, ):
        self.n_pretrain_epoch = n_pretrain_epoch
        self.n_epoch_retrain = n_epoch_retrain
        self.n_epoch_total = n_epoch_total
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pruner = pruner
        self.force_forward_view = force_forward_view
        self.ignore_last_conv = ignore_last_conv
        self.best_result_save_path = best_result_save_path
        self.retrain_if_weight_loaded = retrain_if_weight_loaded


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
                         pruned_best_result_save_path=None,
                         sample_run=None,
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
    # should_train = True
    # if exec_params.best_result_save_path is not None:
    #     if os.path.isfile(exec_params.best_result_save_path):
    #         model.load_state_dict(torch.load(exec_params.best_result_save_path))
    #         if not exec_params.retrain_if_weight_loaded:
    #             should_train = False
    #             test_score = test(model, dataset_params.test_dataset, exec_params.batch_size, use_gpu=use_gpu)
    #             print('Test:\n\tScore: {}'.format(test_score))

    if exec_params.n_pretrain_epoch > 0:
        local_history = train(model, optimizer, dataset_params.train_dataset, exec_params.n_pretrain_epoch,
                              exec_params.batch_size, use_gpu=use_gpu, criterion=criterion,
                              scheduler=scheduler, best_result_save_path=exec_params.best_result_save_path)
        history.append(local_history)
        n_epoch_total = n_epoch_total - exec_params.n_pretrain_epoch

        test_score = test(model, dataset_params.test_dataset, exec_params.batch_size, use_gpu=use_gpu)
        print('Test:\n\tScore: {}'.format(test_score))

    if pruning_params.prune_ratio is not None and exec_params.pruner is not None:
        pruner = exec_params.pruner(model,
                                    sample_run,
                                    exec_params.force_forward_view,
                                    exec_params.ignore_last_conv)

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
            prune_targets = pruner.plan_pruning(max_filters_to_prune_on_iteration)
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

            element_count = pruner.display_pruning_log(prune_targets)

            print("Pruning filters.. ")
            model = model.cpu()
            pruner.prune(prune_targets)

            model = model.cuda()
            # optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.007)
            optimizer = torch.optim.SGD(model.parameters(), exec_params.learning_rate)
            pruner.reset()

            print("Filters pruned {}%".format(100 * float(element_count) / initial_number_of_filters))
            # new_test_score = test(model, dataset_params.test_dataset, exec_params.batch_size, use_gpu=use_gpu)
            # print('Test:\n\tpost prune Score: {}'.format(new_test_score))

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
    end_flops, end_params = profile(model, input_size=(1, 3, 224, 224))
    print("end number of flops: {} \tnumber of params: {}".format(end_flops, end_params))
    print("diff number of flops: {} \tdiff number of params: {}".format((flops - end_flops)/flops, (params - end_params)/params))

    test_score = test(model, dataset_params.test_dataset, exec_params.batch_size, use_gpu=use_gpu)
    print('Final Test:\n\tScore: {}'.format(test_score))

    return history, test_score

