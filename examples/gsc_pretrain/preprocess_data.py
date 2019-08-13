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
Preprocess Google Speech Commands dataset for training
"""

import os
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from audio_transforms import (
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

os.chdir(os.path.dirname(os.path.abspath(__file__)))

EPOCHS = 30
LABELS = tuple(["unknown", "silence", "zero", "one", "two", "three", "four",
                "five", "six", "seven", "eight", "nine"])

DATAPATH = Path("data")
EXTRACTPATH = DATAPATH / "raw"

SEED = 42


def preprocess_wavfiles(folder, wavdata_to_tensor, cachefilepath,
                        silence_percentage=0.0):
    """
    Save a processed dataset from a folder of wav files.

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
    """
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

    x = np.zeros((len(wavdatas), 1, 32, 32), dtype=np.float32)
    for i, d in enumerate(tqdm(wavdatas, leave=False,
                               desc="Processing audio")):
        for xform in wavdata_to_tensor:
            d = xform(d)
        x[i] = d
    y = np.array(ids, dtype=np.int)

    print("Saving preprocessed data to {}".format(cachefilepath))
    np.savez(cachefilepath, x, y)


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    test_wavdata_to_tensor = [
        LoadAudio(),
        FixAudioLength(),
        ToMelSpectrogram(n_mels=32),
        ToTensor("mel_spectrogram", "input"),
        Unsqueeze("input"),
    ]

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

    preprocess_wavfiles(EXTRACTPATH / "valid",
                        test_wavdata_to_tensor,
                        DATAPATH / "gsc_valid.npz")

    for i in range(EPOCHS):
        preprocess_wavfiles(EXTRACTPATH / "train",
                            train_wavdata_to_tensor,
                            DATAPATH / "gsc_train{}.npz".format(i),
                            silence_percentage=0.1)
