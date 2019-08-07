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
Download preprocessed Google Speech Commands dataset
"""

import math
import os
from pathlib import Path

import requests
from tqdm import tqdm


os.chdir(os.path.dirname(os.path.abspath(__file__)))

URL_BASE = "http://public.numenta.com/datasets/google_speech_commands"
FILES = [
    "gsc_train.npz",
    "gsc_valid.npz",
    "gsc_test.npz",
    "gsc_test_noise00.npz",
    "gsc_test_noise05.npz",
    "gsc_test_noise10.npz",
    "gsc_test_noise15.npz",
    "gsc_test_noise20.npz",
    "gsc_test_noise25.npz",
    "gsc_test_noise30.npz",
    "gsc_test_noise35.npz",
    "gsc_test_noise40.npz",
    "gsc_test_noise45.npz",
    "gsc_test_noise50.npz",
]
DATAPATH = Path("data")


def download_files():
    for filename in FILES:
        url = "{}/{}".format(URL_BASE, filename)
        localpath = DATAPATH/filename
        print("Downloading {} to {}".format(url, localpath))
        r = requests.get(url, stream=True)

        total_size = int(r.headers.get("content-length", 0));
        block_size = 1024
        wrote = 0
        with tqdm(total=total_size, unit='B', unit_scale=True, leave=False,
                  desc="Downloading") as pbar:
            with open(localpath, "wb") as f:
                for data in r.iter_content(block_size):
                    wrote = wrote + len(data)
                    f.write(data)
                    pbar.update(len(data))
        if total_size != 0 and wrote != total_size:
            raise requests.exceptions.ConnectionError(
                "Connection to {} failed".format(url))


if __name__ == "__main__":
    os.makedirs(DATAPATH, exist_ok=True)
    download_files()
