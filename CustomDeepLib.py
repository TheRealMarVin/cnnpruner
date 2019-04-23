import os
import random
from shutil import copyfile

import numpy as np
import torch
import time

from sklearn.metrics import accuracy_score
from torch import nn
from torch.autograd import Variable
from torch.utils import model_zoo

from deeplib.datasets import train_valid_loaders
from deeplib.history import History
from deeplib.training import validate
from matplotlib import pyplot as plt
from torch.utils.data import SequentialSampler
from torchvision.transforms import ToTensor


def train(model, optimizer, dataset, n_epoch, batch_size, use_gpu=True, scheduler=None,
          criterion=None, pruner=None, best_result_save_path=None, batch_count=None, should_validate=True):
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
        do_epoch(criterion, model, optimizer, scheduler, train_loader, use_gpu, pruner=pruner, count=batch_count)
        end = time.time()

        if should_validate:
            train_acc, train_loss = validate(model, train_loader, use_gpu)
            val_acc, val_loss = validate(model, val_loader, use_gpu)
            history.save(train_acc, val_acc, train_loss, val_loss, optimizer.param_groups[0]['lr'])
            print(
                'Epoch {} - Train acc: {:.2f} - Val acc: {:.2f} - Train loss: {:.4f} - Val loss: {:.4f} - Training time: {:.2f}s'.format(
                    i, train_acc, val_acc, train_loss, val_loss, end - start))

        if best_result_save_path is not None \
                and val_acc > highest_score:
            highest_score = val_acc
            if os.path.isfile(best_result_save_path):
                copyfile(best_result_save_path, best_result_save_path + ".old")

            basedir = os.path.dirname(best_result_save_path)
            if not os.path.exists(basedir):
                os.makedirs(basedir)
            torch.save(model.state_dict(), best_result_save_path)


    return history


def do_epoch(criterion, model, optimizer, scheduler, train_loader, use_gpu, pruner=None, count=None):
    model.train()
    if scheduler:
        scheduler.step()

    for batch in train_loader:

        inputs, targets = batch
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs)
        targets = Variable(targets)
        optimizer.zero_grad()

        if pruner is not None:
            output = pruner.forward(inputs)
        else:
            output = model(inputs)

        loss = criterion(output, targets)
        loss.backward()

        optimizer.step()
        if pruner is not None:
            pruner.extract_grad(output)

        if count is not None:
            count = count - 1
            if count <= 0:
                break


# fonction de test qui n'override pas la transform
def test(model, test_dataset, batch_size, use_gpu=True):
    sampler = SequentialSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=sampler)

    score, loss = validate(model, test_loader, use_gpu=use_gpu)
    return score


def validate(model, val_loader, use_gpu=True, pruner=None):
    model.train(False)
    true = []
    pred = []
    val_loss = []

    criterion = nn.CrossEntropyLoss()
    model.eval()

    for j, batch in enumerate(val_loader):

        inputs, targets = batch
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
        if pruner is not None:
            output = pruner.forward(inputs)
        else:
            output = model(inputs)

        predictions = output.max(dim=1)[1]

        val_loss.append(criterion(output, targets).item())
        true.extend(targets.data.cpu().numpy().tolist())
        pred.extend(predictions.data.cpu().numpy().tolist())

    model.train(True)
    return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss)


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
