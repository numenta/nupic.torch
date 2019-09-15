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
Run a sparse CNN on the Google Speech Commands dataset
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from audio_transforms import (
    AddNoise,
    ChangeAmplitude,
    ChangeSpeedAndPitchAudio,
    DeleteSTFT,
    FixAudioLength,
    FixSTFTDimension,
    LoadAudio,
    StretchAudioOnSTFT,
    TimeshiftAudioOnSTFT,
    ToMelSpectrogram,
    ToMelSpectrogramFromSTFT,
    ToSTFT,
    ToTensor,
    Unsqueeze,
)
from nupic.torch.models.sparse_cnn import gsc_sparse_cnn, gsc_super_sparse_cnn
from nupic.torch.modules import rezero_weights, update_boost_strength

os.chdir(os.path.dirname(os.path.abspath(__file__)))

LEARNING_RATE = 0.01
LEARNING_RATE_GAMMA = 0.9
MOMENTUM = 0.0
EPOCHS = 30
FIRST_EPOCH_BATCH_SIZE = 4
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 1000
TEST_BATCH_SIZE = 1000
WEIGHT_DECAY = 0.01

LABELS = tuple(["unknown", "silence", "zero", "one", "two", "three", "four",
                "five", "six", "seven", "eight", "nine"])

DATAPATH = Path("data")
EXTRACTPATH = DATAPATH / "raw"


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
    test_wavdata_to_tensor = [
        LoadAudio(),
        FixAudioLength(),
        ToMelSpectrogram(n_mels=32),
        ToTensor("mel_spectrogram", "input"),
        Unsqueeze("input"),
    ]

    valid_dataset = dataset_from_wavfiles(
        EXTRACTPATH / "valid",
        test_wavdata_to_tensor,
        cachefilepath=DATAPATH / "gsc_valid.npz",
    )
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=VALID_BATCH_SIZE)

    train_wavdata_to_tensor = [
        LoadAudio(),
        ChangeAmplitude(),
        ChangeSpeedAndPitchAudio(),
        FixAudioLength(),
        ToSTFT(),
        StretchAudioOnSTFT(),
        TimeshiftAudioOnSTFT(),
        FixSTFTDimension(),
        ToMelSpectrogramFromSTFT(n_mels=32),
        DeleteSTFT(),
        ToTensor("mel_spectrogram", "input"),
        Unsqueeze("input"),
    ]
    sgd = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM,
                    weight_decay=WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.StepLR(sgd, step_size=1,
                                             gamma=LEARNING_RATE_GAMMA)
    for epoch in range(EPOCHS):
        train_dataset = dataset_from_wavfiles(
            EXTRACTPATH / "train",
            train_wavdata_to_tensor,
            cachefilepath=DATAPATH / "gsc_train{}.npz".format(epoch),
            silence_percentage=0.1,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=(FIRST_EPOCH_BATCH_SIZE if epoch == 0
                        else TRAIN_BATCH_SIZE),
            shuffle=True,
        )

        model.apply(update_boost_strength)
        train(model=model, loader=train_loader, optimizer=sgd,
              criterion=F.nll_loss, device=device)
        lr_scheduler.step()
        model.apply(rezero_weights)

        results = test(model=model, loader=valid_loader, criterion=F.nll_loss,
                       device=device)
        print("Epoch {}: {}".format(epoch, results))


def do_noise_test(model, device):
    """
    Test on the noisy data.

    :param model: pytorch model to be tested
    :type model: torch.nn.Module

    :param device:
    :type device: torch.device
    """
    for noise in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        noise_wavdata_to_tensor = [LoadAudio(),
                                   FixAudioLength(),
                                   AddNoise(noise),
                                   ToMelSpectrogram(n_mels=32),
                                   ToTensor("mel_spectrogram", "input"),
                                   Unsqueeze("input")]
        cachefile = "gsc_test_noise{}.npz".format("{:.2f}".format(noise)[2:])
        test_dataset = dataset_from_wavfiles(EXTRACTPATH / "test",
                                             noise_wavdata_to_tensor,
                                             cachefilepath=DATAPATH / cachefile)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=TEST_BATCH_SIZE)
        results = test(model=model, loader=test_loader, criterion=F.nll_loss,
                       device=device)
        print("Noise level: {}, Results: {}".format(noise, results))


def dataset_from_wavfiles(folder, wavdata_to_tensor, cachefilepath,
                          silence_percentage=0.0):
    """
    Get and cache a processed dataset from a folder of wav files.

    :param folder:
    Folder containing wav files in subfolders, for example "./label1/file1.wav"
    :type folder: pathlib.Path

    :param wavdata_to_tensor:
    List of callable objects that create a tensor from a wav file path when
    called in succession.
    :type wavdata_to_tensor: list

    :param cachefilepath:
    Location to save the processed data.
    :type cachefilepath: pathlib.Path

    :param silence_percentage:
    Controls the number of silence wav files that are appended to the dataset.
    :type silence_percentage: float

    :return: torch.utils.data.TensorDataset
    """
    if cachefilepath.exists():
        x, y = np.load(cachefilepath).values()
        x, y = map(torch.tensor, (x, y))
    else:
        label_to_id = {label: i for i, label in enumerate(LABELS)}

        wavdatas = []
        ids = []

        for label in os.listdir(folder):
            if label.startswith("_"):
                continue

            for f in os.listdir(folder / label):
                d = {"path": folder / label / f}
                wavdatas.append(d)
                ids.append(label_to_id[label])

        if silence_percentage > 0.0:
            num_silent = int(len(wavdatas) * silence_percentage)
            for _ in range(num_silent):
                d = {"path": None}
                wavdatas.append(d)
                ids.append(label_to_id["silence"])

        x = torch.zeros(len(wavdatas), 1, 32, 32)
        for i, d in enumerate(tqdm(wavdatas, leave=False,
                                   desc="Processing audio")):
            for xform in wavdata_to_tensor:
                d = xform(d)
            x[i] = d
        y = torch.tensor(ids)

        print("Caching data to {}".format(cachefilepath))
        np.savez(cachefilepath, x.numpy(), y.numpy())

    return torch.utils.data.TensorDataset(x, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supersparse", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=-1)

    args = parser.parse_args()

    if args.seed != -1:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelclass = gsc_super_sparse_cnn if args.supersparse else gsc_sparse_cnn
    model = modelclass(pretrained=args.pretrained).to(device)
    print("Model:")
    print(model)

    if not args.pretrained:
        cache_path = DATAPATH / "cached_model.pth"

        # Option 1: Train model now
        do_training(model, device)
        torch.save(model.state_dict(), cache_path)

        # Option 2: Use previously saved model
        # model.load_state_dict(torch.load(cache_path))

    do_noise_test(model, device)
