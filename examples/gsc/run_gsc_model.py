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

import argparse
import glob
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from nupic.torch.models.sparse_cnn import gsc_sparse_cnn, gsc_super_sparse_cnn
from nupic.torch.modules import rezero_weights, update_boost_strength

from audio_transforms import (ChangeAmplitude, ChangeSpeedAndPitchAudio,
                              FixAudioLength, ToSTFT, StretchAudioOnSTFT,
                              TimeshiftAudioOnSTFT, FixSTFTDimension,
                              ToMelSpectrogramFromSTFT, DeleteSTFT,
                              expand_dims, load_data)


os.chdir(os.path.dirname(os.path.abspath(__file__)))

LEARNING_RATE = 0.01
LEARNING_RATE_GAMMA = 0.9
MOMENTUM = 0.0
EPOCHS = 30
FIRST_EPOCH_BATCH_SIZE = 4
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 1000
TEST_BATCH_SIZE = 1000
REDUCE_LR_ON_PLATEAU = False


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
    for batch_idx, (data, target) in enumerate(tqdm(loader, desc="Train",
                                                    leave=False)):
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
            loss += criterion(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()

    return {"accuracy": total_correct / len(loader.dataset),
            "loss": loss / len(loader.dataset),
            "total_correct": total_correct}



def do_training(model, device):
    """
    Train the model.
    """

    x_valid, y_valid = map(torch.tensor, np.load("data/gsc_valid.npz").values())
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_valid, y_valid),
        batch_size=VALID_BATCH_SIZE)

    x_test, y_test = map(torch.tensor, np.load("data/gsc_test.npz").values())
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=TEST_BATCH_SIZE)

    sgd = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    lr_scheduler = optim.lr_scheduler.StepLR(sgd, step_size=1,
                                             gamma=LEARNING_RATE_GAMMA)

    train_transform = [
        ChangeAmplitude(),
        ChangeSpeedAndPitchAudio(),
        FixAudioLength(),
        ToSTFT(),
        StretchAudioOnSTFT(),
        TimeshiftAudioOnSTFT(),
        FixSTFTDimension(),
        ToMelSpectrogramFromSTFT(n_mels=32),
        DeleteSTFT(),
        expand_dims,
    ]
    train_dir = os.path.join("data", "raw", "train")
    num_files = len(glob.glob("{}/*/*.wav".format(train_dir)))

    for epoch in range(EPOCHS):
        x_train, y_train = zip(*tqdm(load_data(folder=train_dir,
                                               transforms=train_transform),
                                     desc="Processing audio",
                                     total=num_files,
                                     leave=False))

        x_train, y_train  = map(torch.tensor, (x_train, y_train))
        batch_size = (FIRST_EPOCH_BATCH_SIZE if epoch == 0
                      else TRAIN_BATCH_SIZE)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train),
            batch_size=batch_size, shuffle=True)

        train(model=model, loader=train_loader, optimizer=sgd,
              criterion=F.nll_loss, device=device)
        if REDUCE_LR_ON_PLATEAU:
            validation = test(model=model, loader=valid_loader,
                              criterion=F.nll_loss, device=device,
                              desc="Validation")
            lr_scheduler.step(validation["loss"])
        else:
            lr_scheduler.step()
        model.apply(rezero_weights)
        model.apply(update_boost_strength)

        results = test(model=model, loader=test_loader, criterion=F.nll_loss,
                       device=device)
        print("Epoch {}: {}".format(epoch, results))


def do_noise_test(model, device):
    """
    Test on the noisy data.
    """
    for filepath in sorted(glob.glob("data/gsc_test_noise*.npz")):
        suffix = re.match("data/gsc_test_noise(.*).npz", filepath).groups()[0]
        noise = float("0.{}".format(suffix))
        x_test, y_test = map(torch.tensor, np.load(filepath).values())
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)
        results = test(model=model, loader=test_loader, criterion=F.nll_loss,
                       device=device)

        print("Noise level: {}, Results: {}".format(noise, results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supersparse", action="store_true")
    parser.add_argument("--pretrained", action="store_true")

    args = parser.parse_args()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelclass = (gsc_super_sparse_cnn if args.supersparse else gsc_sparse_cnn)
    model = modelclass(pretrained=args.pretrained).to(device)
    print("Model:")
    print(model)

    if not args.pretrained:
        do_training(model, device)

    do_noise_test(model, device)
