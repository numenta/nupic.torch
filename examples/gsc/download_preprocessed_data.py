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

import os
import shutil
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))

FILENAME = "gsc_preprocessed_v0.01.tar.gz"
URL = "http://public.numenta.com/datasets/google_speech_commands/{}".format(FILENAME)
DATAPATH = Path("data")
TARFILEPATH = DATAPATH / FILENAME


def download_tarball():
    tmppath = str(TARFILEPATH) + ".temp"
    print("Downloading {} to {}".format(URL, TARFILEPATH))
    r = requests.get(URL, stream=True)
    r.raise_for_status()

    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024
    wrote = 0
    with tqdm(total=total_size, unit="B", unit_scale=True, leave=False,
              desc="Downloading") as pbar:
        with open(tmppath, "wb") as f:
            for data in r.iter_content(block_size):
                wrote = wrote + len(data)
                f.write(data)
                pbar.update(len(data))
    if total_size != 0 and wrote != total_size:
        raise requests.exceptions.ConnectionError(
            "Connection to {} failed".format(URL))
    else:
        shutil.move(tmppath, TARFILEPATH)


def extract_tarball():
    print("Extracting {} to {}".format(TARFILEPATH, DATAPATH))
    with tarfile.open(TARFILEPATH) as tar:
        # This is slow to count.
        tot = 42  # len(list(tar.getnames()))
        tar.extractall(DATAPATH,
                       members=tqdm(tar, desc="Extracting", total=tot,
                                    unit="file", unit_scale=True, leave=False))


if __name__ == "__main__":
    os.makedirs(DATAPATH, exist_ok=True)
    download_tarball()
    extract_tarball()
