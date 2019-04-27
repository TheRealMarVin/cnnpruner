import os
import torch

from torch import nn
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from Pruner.FilterPruner import FilterPruner
from deeplib_ext.CustomDeepLib import train, test
from FileHelper import load_obj, save_obj
from ModelHelper import total_num_filters
from deeplib_ext.MultiHistory import MultiHistory
from deeplib_ext.history import History
from models.AlexNetSki import alexnetski

from thop_ext.profile import profile

# TODO check this one!!! https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
# and this: https://github.com/fg91/visualizing-cnn-feature-maps/blob/master/Calculate_mean_activation_per_filter_in_specific_layer_given_an_image.ipynb


def common_training_code(model,
                         pruned_save_path=None,
                         best_result_save_path=None,
                         pruned_best_result_save_path=None,
                         retrain_if_weight_loaded=False,
                         sample_run=None,
                         reuse_cut_filter=False,
                         max_percent_per_iteration=0.1,
                         prune_ratio=0.3,
                         n_epoch=10,
                         n_epoch_retrain=3,
                         n_epoch_total=20):
    test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = CIFAR10("C:/dev/data/cifar10/", train=True, transform=test_transform, download=True)
    test_dataset = CIFAR10("C:/dev/data/cifar10/", train=False, transform=test_transform, download=True)

    # display_sample_data(train_dataset)
    # display_sample_data(test_dataset)
    model.cuda()

    flops, params = profile(model, input_size=(1, 3, 224, 224))
    print("number of flops: {} \tnumber of params: {}".format(flops, params))

    use_gpu = True
    batch_size = 16
    learning_rate = 0.01

    history = History()

    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
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
    if should_train:
        local_history = train(model, optimizer, train_dataset, n_epoch,
                              batch_size, use_gpu=use_gpu, criterion=criterion,
                              scheduler=scheduler, best_result_save_path=best_result_save_path)
        history.append(local_history)
        n_epoch_total = n_epoch_total - n_epoch
        # history.display()

    test_score = test(model, test_dataset, batch_size, use_gpu=use_gpu)
    print('Test:\n\tScore: {}'.format(test_score))

    ###
    #TODO maybe put the loop content in a function that looks terrible now
    if prune_ratio is not None:
        pruner = FilterPruner(model, sample_run)
        number_of_filters = total_num_filters(model)
        filter_to_prune = int(number_of_filters * prune_ratio)
        max_filters_to_prune_on_iteration = int(number_of_filters * max_percent_per_iteration)
        if filter_to_prune < max_filters_to_prune_on_iteration:
            max_filters_to_prune_on_iteration = filter_to_prune
        iterations = (filter_to_prune//max_filters_to_prune_on_iteration)
        print("{} iterations to reduce {:2.2f}% filters".format(iterations, prune_ratio*100))

        for param in model.parameters():
            param.requires_grad = True

        for iteration_idx in range(iterations):
            print("Perform pruning iteration: {}".format(iteration_idx))
            pruner.reset()

            prune_targets = None
            if reuse_cut_filter:
                prune_targets = load_obj("filters_dic")

            if prune_targets is None:
                train(model, optimizer, train_dataset, 1, batch_size, use_gpu=use_gpu, criterion=criterion,
                      scheduler=scheduler, pruner=pruner, batch_count=1, should_validate=False)

                pruner.normalize_layer()

                prune_targets = pruner.plan_prunning(max_filters_to_prune_on_iteration)
                if reuse_cut_filter:
                    save_obj(prune_targets, "filters_dic")

            pruner.display_pruning_log(prune_targets)

            print("Pruning filters.. ")
            model = model.cpu()
            pruner.prune(prune_targets)

            model = model.cuda()
            # optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.007)
            optimizer = torch.optim.SGD(model.parameters(), learning_rate)
            pruner.reset()

            print("Filters pruned {}%".format(100 - (100 * float(total_num_filters(model)) / number_of_filters)))
            new_test_score = test(model, test_dataset, batch_size, use_gpu=use_gpu)
            print('Test:\n\tpost prune Score: {}'.format(new_test_score))

            basedir = os.path.dirname(pruned_save_path)
            if not os.path.exists(basedir):
                os.makedirs(basedir)
            torch.save(model, pruned_save_path)

            print("Fine tuning to recover from prunning iteration.")
            local_history = train(model, optimizer, train_dataset, n_epoch_retrain, batch_size, use_gpu=use_gpu,
                                  criterion=None, scheduler=scheduler, pruner=None,
                                  best_result_save_path=pruned_best_result_save_path)
            n_epoch_total = n_epoch_total - n_epoch_retrain
            history.append(local_history)
            # local_history.display()
            test_score = test(model, test_dataset, batch_size, use_gpu=use_gpu)
            print('Test pruning iteration :{}\n\tScore: {}'.format(iteration_idx, test_score))
    ###

    test_score = test(model, test_dataset, batch_size, use_gpu=use_gpu)
    print('Test Fin :\n\tScore: {}'.format(test_score))

    if n_epoch_total > 0:
        local_history = train(model, optimizer, train_dataset, n_epoch_total,
                              batch_size, use_gpu=use_gpu, criterion=criterion,
                              scheduler=scheduler, best_result_save_path=pruned_best_result_save_path)
        history.append(local_history)

    basedir = os.path.dirname(pruned_save_path)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    torch.save(model, pruned_save_path)

    history.display()
    flops, params = profile(model, input_size=(1, 3, 224, 224))
    print("end number of flops: {} \tnumber of params: {}".format(flops, params))

    return history


def exec_alexnet(max_percent_per_iteration=0.1, prune_ratio=0.3, n_epoch=10):
    print("***alexnet")
    model = alexnetski(num_classes=10)
    model.cuda()

    history = common_training_code(model, pruned_save_path="saved/alex{}/PrunedAlexnet.pth".format(prune_ratio),
                                   # best_result_save_path="saved/alex{}/alexnet.pth".format(prune_ratio),
                                   pruned_best_result_save_path="saved/alex{}/alexnet_pruned.pth".format(prune_ratio),
                                   sample_run=torch.zeros([1, 3, 224, 224]),
                                   reuse_cut_filter=False,
                                   max_percent_per_iteration=max_percent_per_iteration,
                                   prune_ratio=prune_ratio,
                                   n_epoch=n_epoch,
                                   n_epoch_retrain=2,
                                   n_epoch_total=20)

    return history


def exec_dense_net(max_percent_per_iteration=0.1, prune_ratio=0.3, n_epoch=10):
    print("***densenet121")

    model = models.densenet121(num_classes=10)
    model.cuda()

    history = common_training_code(model, pruned_save_path="saved/resnet18/Prunedresnet.pth,",
                                   # best_result_save_path="saved/resnet18/resnet18.pth",
                                   pruned_best_result_save_path="saved/resnet18/resnet18_pruned.pth",
                                   sample_run=torch.zeros([1, 3, 224, 224]),
                                   reuse_cut_filter=False,
                                   max_percent_per_iteration=max_percent_per_iteration,
                                   prune_ratio=prune_ratio,
                                   n_epoch=n_epoch,
                                   n_epoch_retrain=2,
                                   n_epoch_total=20)
    return history


def exec_vgg16(max_percent_per_iteration=0.1, prune_ratio=0.3, n_epoch=10):
    print("***vgg16")

    model = models.vgg16(num_classes=10)
    model.cuda()

    history = common_training_code(model, pruned_save_path="saved/resnet18/Prunedresnet.pth,",
                                   # best_result_save_path="saved/resnet18/resnet18.pth",
                                   pruned_best_result_save_path="saved/resnet18/resnet18_pruned.pth",
                                   sample_run=torch.zeros([1, 3, 224, 224]),
                                   reuse_cut_filter=False,
                                   max_percent_per_iteration=max_percent_per_iteration,
                                   prune_ratio=prune_ratio,
                                   n_epoch=n_epoch,
                                   n_epoch_retrain=2,
                                   n_epoch_total=20)
    return history


def exec_resnet18(max_percent_per_iteration=0.1, prune_ratio=0.3, n_epoch=10):
    print("***resnet 18")

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    model.cuda()

    history = common_training_code(model, pruned_save_path="saved/resnet18/Prunedresnet.pth,",
                                   # best_result_save_path="saved/resnet18/resnet18.pth",
                                   pruned_best_result_save_path="saved/resnet18/resnet18_pruned.pth",
                                   sample_run=torch.zeros([1, 3, 224, 224]),
                                   reuse_cut_filter=False,
                                   max_percent_per_iteration=max_percent_per_iteration,
                                   prune_ratio=prune_ratio,
                                   n_epoch=n_epoch,
                                   n_epoch_retrain=2,
                                   n_epoch_total=20)
    return history


def exec_resnet34(max_percent_per_iteration=0.1, prune_ratio=0.3, n_epoch=10):
    print("***resnet 34")

    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    model.cuda()

    history = common_training_code(model, pruned_save_path="saved/resnet34/Prunedresnet.pth,",
                                   # best_result_save_path="saved/resnet34/resnet18.pth",
                                   pruned_best_result_save_path="saved/resnet34/resnet34_pruned.pth",
                                   sample_run=torch.zeros([1, 3, 224, 224]),
                                   reuse_cut_filter=False,
                                   max_percent_per_iteration=max_percent_per_iteration,
                                   prune_ratio=prune_ratio,
                                   n_epoch=n_epoch,
                                   n_epoch_retrain=2,
                                   n_epoch_total=20)
    return history


def exec_resnet50(max_percent_per_iteration=0.1, prune_ratio=0.3, n_epoch=10):
    print("***resnet 50")

    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    model.cuda()

    history = common_training_code(model, pruned_save_path="saved/resnet50/Prunedresnet.pth,",
                                   # best_result_save_path="saved/resnet50/resnet18.pth",
                                   pruned_best_result_save_path="saved/resnet50/resnet50_pruned.pth",
                                   sample_run=torch.zeros([1, 3, 224, 224]),
                                   reuse_cut_filter=False,
                                   max_percent_per_iteration=max_percent_per_iteration,
                                   prune_ratio=prune_ratio,
                                   n_epoch=n_epoch,
                                   n_epoch_retrain=2,
                                   n_epoch_total=20)
    return history


def run_startegy_prune_compare():
    multi_history = MultiHistory()
    h = exec_resnet18(max_percent_per_iteration=0.0, prune_ratio=None, n_epoch=10)
    multi_history.append_history("Resnet 18-0", h)
    h = exec_resnet18(max_percent_per_iteration=0.2, prune_ratio=0.2, n_epoch=10)
    multi_history.append_history("Resnet 18-20", h)
    # h = exec_resnet34()
    # multi_history.append_history("Resnet 34", h)
    # h = exec_resnet50()
    # multi_history.append_history("Resnet 50", h)
    h = exec_alexnet(max_percent_per_iteration=0.0, prune_ratio=None, n_epoch=20)
    multi_history.append_history("Alexnet 0", h)
    h = exec_alexnet(max_percent_per_iteration=0.2, prune_ratio=0.2, n_epoch=20)
    multi_history.append_history("Alexnet 20", h)

    h = exec_vgg16(max_percent_per_iteration=0.0, prune_ratio=None, n_epoch=10)
    multi_history.append_history("vgg16 0", h)
    h = exec_vgg16(max_percent_per_iteration=0.2, prune_ratio=0.2, n_epoch=10)
    multi_history.append_history("vgg16 20", h)

    # h = exec_dense_net(max_percent_per_iteration=0.0, prune_ratio=None, n_epoch=10)
    # multi_history.append_history("densenet121 0", h)
    # h = exec_dense_net(max_percent_per_iteration=0.2, prune_ratio=0.2, n_epoch=10)
    # multi_history.append_history("densenet121 20", h)



    save_obj(multi_history, "history_compare")
    multi_history.display_single_key(History.VAL_ACC_KEY)


def run_alex_prune_compare():
    multi_history = MultiHistory()
    h = exec_alexnet(max_percent_per_iteration=0.0, prune_ratio=None, n_epoch=20)
    multi_history.append_history("Alexnet 0%", h)
    h = exec_alexnet(max_percent_per_iteration=0.05, prune_ratio=0.1)
    multi_history.append_history("Alexnet 10%", h)
    h = exec_alexnet(max_percent_per_iteration=0.15, prune_ratio=0.3)
    multi_history.append_history("Alexnet 30%", h)
    # h = exec_alexnet(max_percent_per_iteration=0.1, prune_ratio=0.3)
    # multi_history.append_history("Alexnet 30%-3", h)
    h = exec_alexnet(max_percent_per_iteration=0.25, prune_ratio=0.5)
    multi_history.append_history("Alexnet 50%", h)
    h = exec_alexnet(max_percent_per_iteration=0.25, prune_ratio=0.75)
    multi_history.append_history("Alexnet 75%", h)
    save_obj(multi_history, "history_alex")
    multi_history.display_single_key(History.VAL_ACC_KEY)


if __name__ == '__main__':
    # multi_history = MultiHistory()
    # h = exec_dense_net(max_percent_per_iteration=0.3, prune_ratio=0.3, n_epoch=1)
    # multi_history.append_history("densenet121 20", h)

    run_alex_prune_compare()
    run_startegy_prune_compare()
