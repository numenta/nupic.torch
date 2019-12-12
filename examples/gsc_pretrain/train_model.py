# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
Train a sparse CNN on the Google Speech Commands dataset
"""

import argparse
import copy
import hashlib
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from nupic.torch.models.sparse_cnn import GSCSparseCNN, GSCSuperSparseCNN
from nupic.torch.modules import rezero_weights, update_boost_strength

os.chdir(os.path.dirname(os.path.abspath(__file__)))


SEED = 42
LEARNING_RATE = 0.01
LEARNING_RATE_GAMMA = 0.9
MOMENTUM = 0.0
EPOCHS = 30
FIRST_EPOCH_BATCH_SIZE = 4
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 1000
TEST_BATCH_SIZE = 1000
WEIGHT_DECAY = 0.01

DATAPATH = Path("data")


def train(model, loader, optimizer, criterion, device):
    """
    Train the model using given dataset loader.
    Called on every epoch.

    :param model: pytorch model to be trained
    :type model: torch.nn.Module
    :param loader: DataLoader configured for the epoch.
    :type loader: :class:`torch.utils.data.DataLoader`
    :param optimizer: Optimizer object used to train the model.
    :type optimizer: :class:`torch.optim.Optimizer`
    :param criterion: loss function to use
    :type criterion: function
    :param device:
    :type device: :class:`torch.device`
    """
    model.train()
    for data, target in tqdm(loader, desc="Train", leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test(model, loader, criterion, device, desc="Test"):
    """
    Evaluate trained model using given dataset loader.
    Called on every epoch.

    :param model: Pretrained pytorch model
    :type model: torch.nn.Module
    :param loader: DataLoader configured for the epoch.
    :type loader: :class:`torch.utils.data.DataLoader`
    :param criterion: loss function to use
    :type criterion: function
    :param device:
    :type device: :class:`torch.device`
    :param desc: Description for progress bar
    :type desc: str
    :return: Dict with "accuracy", "loss" and "total_correct"
    """
    model.eval()
    loss = 0
    total_correct = 0
    with torch.no_grad():
        for data, target in tqdm(loader, desc=desc, leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            loss += criterion(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()

    return {"accuracy": total_correct / len(loader.dataset),
            "loss": loss / len(loader.dataset),
            "total_correct": total_correct}


def do_training(model, device):
    """
    Train the model.

    :param model: pytorch model to be trained
    :type model: torch.nn.Module

    :param device:
    :type device: torch.device
    """

    valid_dataset = preprocessed_dataset(DATAPATH / "gsc_valid.npz")
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=VALID_BATCH_SIZE)

    sgd = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM,
                    weight_decay=WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.StepLR(sgd, step_size=1,
                                             gamma=LEARNING_RATE_GAMMA)
    best_model = None
    best_results = {}
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(EPOCHS):
        train_dataset = preprocessed_dataset(
            DATAPATH / "gsc_train{}.npz".format(epoch)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
        )

        model.apply(update_boost_strength)
        train(model=model, loader=train_loader, optimizer=sgd,
              criterion=F.nll_loss, device=device)
        lr_scheduler.step()
        model.apply(rezero_weights)

        results = test(model=model, loader=valid_loader, criterion=F.nll_loss,
                       device=device)

        # Save best model
        if results["accuracy"] > best_acc:
            best_acc = results["accuracy"]
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            best_results.update(results)

        print("Epoch {}: {}".format(epoch, results))

    print("Best model: {}: {}".format(best_epoch, best_results))
    return best_model


def preprocessed_dataset(filepath):
    """
    Get a processed dataset

    :param cachefilepath:
    Path to the processed data.
    :type cachefilepath: pathlib.Path

    :return: torch.utils.data.TensorDataset
    """
    x, y = np.load(filepath).values()
    x, y = map(torch.tensor, (x, y))

    return torch.utils.data.TensorDataset(x, y)


if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--supersparse", action="store_true")
    args = parser.parse_args()

    if args.supersparse:
        modelclass = GSCSuperSparseCNN
        savepath_fmt = str(DATAPATH / "gsc_super_sparse_cnn-{}.pth")
    else:
        modelclass = GSCSparseCNN
        savepath_fmt = str(DATAPATH / "gsc_sparse_cnn-{}.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = modelclass().to(device)
    print("Training model:")
    print(model)
    model = do_training(model, device)

    tmp = DATAPATH / "tmp.pth"
    model.cpu()
    torch.save(model.state_dict(), tmp)

    # Compute final filename
    with open(tmp, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:8]
    savepath = savepath_fmt.format(sha)
    print("Saving {}".format(savepath))
    shutil.move(tmp, savepath)
