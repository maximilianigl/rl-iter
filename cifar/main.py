'''Train CIFAR10 with PyTorch.'''
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
import copy

from models import *
from utils import progress_bar
import platform
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred import SETTINGS

SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('cifar')
ex.add_config('./default.yaml')
ex.observers.append(FileStorageObserver('../db'))

if platform.system() == "Darwin":
    print("Running on laptop, setting small batch size")
    batch_size_train = batch_size_test = 16
    debug = True
else:
    batch_size_train = 128
    batch_size_test = 100
    debug = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@ex.config
def config(annealing):
    if not annealing["every_n_epochs"]:
        annealing["every_n_epochs"] = annealing["duration"]
    assert annealing["every_n_epochs"] <= annealing["duration"]
    if annealing["add_fraction"] is None:
        annealing["add_fraction"] = (
            (1 - annealing["start_fraction"]) / annealing["duration"]
        ) * annealing["every_n_epochs"]

    if annealing["never_same"] is None:
        if annealing["type"] in ["consistent", "anneal_consistent"]:
            annealing["never_same"] = True
        else:
            annealing["never_same"] = False




@ex.capture
def test(epoch, net, dataloader, criterion, _run,
         log_prefix=""):
    # global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    ib_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, kl = net(inputs)
            kl = kl.sum()
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            ib_loss += kl.item()

            if debug:
                string = '%s Loss: %.3f | Acc: %.3f%% | Kl: %.3f (%d/%d)' % (
                    log_prefix,
                    test_loss / (batch_idx + 1),
                    100. * correct / total,
                    kl.item(),
                    correct,
                    total)
                print(batch_idx, len(dataloader), string)
            else:
                string = '%s Loss: %.3f | Acc: %.3f%% | Kl: %.3f (%d/%d)' % (
                    log_prefix,
                    test_loss / (batch_idx + 1),
                    100. * correct / total,
                    kl.item(),
                    correct,
                    total)
                progress_bar(batch_idx, len(dataloader), string)

    _run.log_scalar(log_prefix + "test_loss", test_loss / len(dataloader), epoch)
    _run.log_scalar(log_prefix + "test_acc", 100. * correct / total, epoch)
    _run.log_scalar(log_prefix + "test_ib", ib_loss / len(dataloader), epoch)

@ex.capture
def train(epoch, net, dataloader, criterion, optimizer, vib_beta, _run,
          log_prefix=""):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    ib_loss = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, kl = net(inputs)
        kl = kl.sum()
        loss = criterion(outputs, targets)
        total_loss = loss + vib_beta * kl
        total_loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)

        if len(targets.size()) > 1:
            # This is if I replaced targets with probabilities from old teacher
            _, targets = targets.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        ib_loss += kl.item()

        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        if debug:
            string = '%s Loss: %.3f | Acc: %.3f%% | Kl: %.3f (%d/%d) ' % (
                log_prefix,
                train_loss / (batch_idx + 1),
                100. * correct / total,
                kl.item(),
                correct, total,
            )
            print(batch_idx, len(dataloader), string)
        else:
            string = '%s Loss: %.3f | Acc: %.3f%% | Kl: %.3f (%d/%d)' % (
                log_prefix,
                train_loss / (batch_idx + 1),
                100. * correct / total,
                kl.item(),
                correct, total)
            progress_bar(batch_idx, len(dataloader), string)
    _run.log_scalar(log_prefix + "train_loss", train_loss / len(dataloader), epoch)
    _run.log_scalar(log_prefix + "train_acc", 100. * correct / total, epoch)
    _run.log_scalar(log_prefix + "train_ib", ib_loss / len(dataloader), epoch)


def save_model(model, _run, name):
    filename = f"{name}_{_run._id}.pyt"
    db_filename = f"{name}.pyt"
    torch.save(model, filename)
    _run.add_artifact(filename, name=db_filename)
    os.remove(filename)

@ex.capture
def randomize_targets(trainset, annealing):
    original_targets = copy.deepcopy(trainset.targets)
    num_classes = 10 #max(trainset.targets) + 1,
    trainset.targets = np.random.choice(
        num_classes,
        len(trainset.targets)
    )
    if annealing['never_same']:
        # Make sure that _no_ target is the correct one, not even randomly
        # trainset.targets = np.array(trainset.targets)
        # idx = np.array(original_targets) == trainset.targets
        idx = original_targets == trainset.targets
        n_idx = int(sum(idx))
        while n_idx > 0:
            print(f"Reshuffling {n_idx} entries")
            random_subset = np.random.choice(
                num_classes, n_idx)
            trainset.targets[idx] = random_subset
            # idx = np.array(original_targets) == trainset.targets
            idx = original_targets == trainset.targets
            n_idx = int(sum(idx))
        # trainset.targets = list(trainset.targets)

def flip_one_permutation(targets):
    print(f"Flipping permutation from {targets}")
    num_classes = 10
    ia, ib = np.random.choice(num_classes, 2, replace=False)
    targets[ia], targets[ib] = targets[ib], targets[ia]
    print(f"to {targets}")

@ex.capture
def get_current_trainloader(indexes_correct, permutation_targets, trainset_full,
                            trainset_full_wrong_targets,
                            annealing):

    if np.all(indexes_correct):
        return torch.utils.data.DataLoader(
            copy.deepcopy(trainset_full),
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=2)

    num_classes = 10
    if annealing["type"] == "size":
        trainset_current = copy.deepcopy(trainset_full)

        mult_factor = len(trainset_full) // sum(indexes_correct)
        # Subsample data
        assert len(trainset_current.data.shape) == 4
        trainset_current.data = np.tile(
            trainset_current.data[indexes_correct],
            reps=(mult_factor, 1, 1, 1),  # only tile the batch dimension
        )
        # trainset_current.targets = list(
        #     np.array(trainset_current.targets)[indexes_correct]
        # ) * mult_factor
        trainset_current.targets = np.tile(
            trainset_current.targets[indexes_correct],
            reps=(mult_factor, ),
        )


        return torch.utils.data.DataLoader(trainset_current, batch_size=batch_size_train, shuffle=True, num_workers=2)

    if annealing["type"] == "random":
        # And randomize targets if needed, otherwise they will always be "wrong" in the same way
        trainset_current = copy.deepcopy(trainset_full_wrong_targets)
        randomize_targets(trainset_current)
    elif annealing["type"] == "consistent":
        trainset_current = copy.deepcopy(trainset_full_wrong_targets)
    elif annealing["type"] == "permute":
        trainset_current = copy.deepcopy(trainset_full)
        # trainset_current.targets = np.array(trainset_current.targets)

        for label, perm_target in enumerate(permutation_targets):
            trainset_current.targets = np.where(
                trainset_current.targets == label,
                perm_target,
                trainset_current.targets
            )

        # trainset_current.targets = list(trainset_current.targets)

    elif annealing["type"] in ["anneal_zero", "anneal_consistent", "anneal_random"]:
        trainset_current = copy.deepcopy(trainset_full)

        frac_correct = sum(indexes_correct) / len(indexes_correct)
        frac_wrong = 1 - frac_correct  # 1-> 0
        max_target_wrong = math.ceil(frac_wrong * 10)  # Goes from 10 -> 0

        # Replace all targets < max_target_wrong with target=0 or from wrong dataset
        # trainset_current.targets = np.array(trainset_current.targets)

        if annealing["type"] == "anneal_zero":
            trainset_current.targets = np.where(
                trainset_current.targets<max_target_wrong,
                0, trainset_current.targets
            )
        elif annealing["type"] == "anneal_consistent":
            trainset_current.targets = np.where(
                trainset_current.targets<max_target_wrong,
                trainset_full_wrong_targets.targets, trainset_current.targets
            )
        elif annealing["type"] == "anneal_random":
            random_targets = np.random.choice(
                num_classes,
                len(trainset_current.targets)
            )
            trainset_current.targets = np.where(
                trainset_current.targets<max_target_wrong,
                random_targets, trainset_current.targets
            )

        # trainset_current.targets = list(trainset_current.targets)

    elif annealing["type"] is None:
        # Create with correct targets
        trainset_current = copy.deepcopy(trainset_full)
    else:
        raise NotImplementedError("Wrong annealing.type")

    # Selective copy correct ones
    # trainset_current.targets = np.array(trainset_current.targets)
    trainset_current.targets[indexes_correct] = np.array(trainset_full.targets)[indexes_correct]
    # trainset_current.targets = list(trainset_current.targets)
    return torch.utils.data.DataLoader(trainset_current, batch_size=batch_size_train, shuffle=True, num_workers=2)

def add_n_True(full_array, n):
    indexes_false = np.nonzero(np.invert(full_array))[0]
    try:
        indexes_to_change = np.random.choice(indexes_false, n, replace=False)
    except ValueError:
        indexes_to_change = np.random.choice(indexes_false, len(indexes_false), replace=False)

    full_array[indexes_to_change] = True


@ex.capture
def test_encoder(net, trainloader, testloader, train_criterion, test_criterion,
                 frozen_test_epochs, lr, _run):
    print("Creating copy with frozen encoder...")
    log_prefix = "frozen/"
    # frozen_net = copy.deepcopy(net.module)
    frozen_net = copy.deepcopy(net)
    frozen_net.reset_last_layer()
    frozen_net.freeze()
    print("Move frozen_net to device")
    frozen_net = frozen_net.to(device)
    if device == 'cuda':
        # frozen_net = torch.nn.DataParallel(frozen_net)
        cudnn.benchmark = True

    print("Move to device")
    optimizer = optim.SGD(frozen_net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    print("Training frozen encoder net...")

    for epoch in range(frozen_test_epochs):

        train(epoch, frozen_net, trainloader, train_criterion, optimizer, log_prefix=log_prefix)
        test(epoch, frozen_net, testloader, test_criterion, log_prefix=log_prefix)

    save_model(net, _run, "model_frozen_final")
    print("... Done. Going back to normal training.")

def replace_targets_with_teacher(net, trainset_full):
    bs = 100

    net.eval()
    dataloader = torch.utils.data.DataLoader(
        copy.deepcopy(trainset_full),
        batch_size=bs,
        shuffle=False,
        num_workers=2)

    new_targets = np.zeros(shape=(50000, 10), dtype=np.float32)
    print("Setting new targets")

    for batch_idx, (inputs, _) in enumerate(dataloader):
        current_idx = batch_idx * bs
        with torch.no_grad():
            inputs = inputs.to(device)
            # As with NLLLoss, the input given is expected to contain
            # log-probabilities and is not restricted to a 2D Tensor.
            # The targets are given as probabilities (i.e. without taking the logarithm).
            targets, _ = net(inputs)
            # targets = torch.nn.functional.log_softmax(targets, dim=1).cpu().numpy()
            targets = torch.nn.functional.softmax(targets, dim=1).cpu().numpy()
        new_targets[current_idx : current_idx + bs] = targets

    trainset_full.targets = new_targets

    return trainset_full


kl_div_crit = nn.KLDivLoss()
def kl_div_criterion(inputs, targets):
    inputs = torch.nn.functional.log_softmax(inputs, dim=1)
    return kl_div_crit(inputs, targets)

ce_crit = nn.CrossEntropyLoss()
def onehot_criterion(inputs, targets):
    _, targets = targets.max(1)
    return ce_crit(inputs, targets)


@ex.automain
def main(seed, lr, epochs, use_bn, use_vib, self_distillation, self_distillation_epochs,
         self_dist_crit, annealing, frozen_test_epochs, _run):
    np.random.seed(seed)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainset_full.targets = np.array(trainset_full.targets)
    trainloader_full = torch.utils.data.DataLoader(trainset_full, batch_size=batch_size_train, shuffle=True, num_workers=2)

    trainset_full_wrong_targets = copy.deepcopy(trainset_full)
    trainset_full_wrong_targets.targest = np.array(trainset_full_wrong_targets.targets)
    randomize_targets(trainset_full_wrong_targets)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = PreActResNet18(use_bn=use_bn, use_vib=use_vib)
    print("Move to device")
    net = net.to(device)
    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    train_criterion = test_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    indexes_correct= np.array([False] * len(trainset_full), dtype=bool)
    permutation_targets = np.random.choice(10, 10, replace=False)

    # How many correct/available data to start with
    start_number = math.ceil(len(trainset_full) * annealing["start_fraction"])

    # How many correct/available data to add every n iterations
    add_number = math.ceil(len(trainset_full) * annealing["add_fraction"])

    # Add `start_number` to the list
    add_n_True(indexes_correct, start_number)

    # Training
    # for epoch in range(start_epoch, start_epoch+200):
    for epoch in range(epochs):

        # Add correct indices
        if epoch == annealing['duration']:
            indexes_correct = np.array([True] * len(trainset_full), dtype=bool)
            permutation_targets = np.array(list(range(10)))
        elif (epoch > 0 and epoch < annealing['duration']
                and epoch % int(annealing["every_n_epochs"]) == 0):
            print("Updating correct labels")
            add_n_True(indexes_correct, add_number)
            flip_one_permutation(permutation_targets)


        # Save model at end of annealing
        if epoch > 0 and epoch == annealing["duration"]:
            save_model(net, _run, "model_end_nonstat")
            if frozen_test_epochs > 0:
                test_encoder(net, trainloader_full, testloader, train_criterion, test_criterion)

        # Self-distillation
        if (self_distillation != 0 and epoch >= self_distillation
                and (epoch - self_distillation) % self_distillation_epochs == 0):
            assert self_distillation > annealing['duration']
            assert not use_vib

            # Set new targets based on teacher output
            trainset_full = replace_targets_with_teacher(net, trainset_full)

            # New Criterion
            if self_dist_crit == 'kl':
                train_criterion = kl_div_criterion
            elif self_dist_crit == 'onehot':
                train_criterion = onehot_criterion
            else:
                raise NotImplementedError()

            # Reset model weights:
            student = PreActResNet18(use_bn=use_bn, use_vib=use_vib).to(device)
            net.load_state_dict(student.state_dict())
            del student


        print("Updating trainloader")
        trainloader_current = get_current_trainloader(
            indexes_correct, permutation_targets,
            trainset_full, trainset_full_wrong_targets
        )

        _run.log_scalar("frac_correct", sum(indexes_correct)/len(indexes_correct), epoch)

        train(epoch, net, trainloader_current, train_criterion, optimizer)
        test(epoch, net, testloader, test_criterion)

    save_model(net, _run, "model_final")


