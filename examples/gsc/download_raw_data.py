"""
Download, extract, and organize Google Speech Commands dataset
"""

import math
import os
from pathlib import Path
import re
import shutil
import tarfile

import requests
from tqdm import tqdm


os.chdir(os.path.dirname(os.path.abspath(__file__)))

FILENAME = "speech_commands_v0.01.tar.gz"
URL = "http://download.tensorflow.org/data/{}".format(FILENAME)
TARFILEPATH = Path("data")/FILENAME
EXTRACTPATH = Path("data")/"raw"


def download_tarball():
    print("Downloading {} to {}".format(URL, TARFILEPATH))
    r = requests.get(URL, stream=True)

    total_size = int(r.headers.get("content-length", 0));
    block_size = 1024
    wrote = 0

    with tqdm(total=total_size, unit='B', unit_scale=True, leave=False,
              desc="Downloading") as pbar:
        with open(TARFILEPATH, "wb") as f:
            for data in r.iter_content(block_size):
                wrote = wrote + len(data)
                f.write(data)
                pbar.update(len(data))

    if total_size != 0 and wrote != total_size:
        raise requests.exceptions.ConnectionError(
            "Connection to {} failed".format(URL))


def extract_tarball():
    print("Extracting {} to {}".format(TARFILEPATH, EXTRACTPATH))
    with tarfile.open(TARFILEPATH) as tar:
        # This is slow to count.
        tot = 64764 # len(list(tar.getnames()))
        tar.extractall(EXTRACTPATH,
                       members=tqdm(tar, desc="Extracting", total=tot,
                                    unit="file", unit_scale=True))


def organize_files():
    print("Organizing files into train, validation, and test folders")
    used_categories = ["zero", "one", "two", "three", "four", "five", "six",
                       "seven", "eight", "nine"]

    # Move unused categories into "unused" folder
    files = os.listdir(EXTRACTPATH) # Get list before creating "unused" folder
    destdir = EXTRACTPATH/"unused"
    os.mkdir(destdir)
    for name in files:
        fullpath = EXTRACTPATH/name
        if os.path.isdir(fullpath):
            if not (name in used_categories or name == "_background_noise_"):
                newpath = destdir/name
                shutil.move(fullpath, newpath)

    # Set aside validation and test sets
    for listfile, dest in [("validation_list.txt", "valid"),
                           ("testing_list.txt", "test")]:
        destdir = EXTRACTPATH/dest
        os.mkdir(destdir)
        for category in used_categories:
            os.mkdir(destdir/category)
        with open(EXTRACTPATH/listfile, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                category = re.match("(.*)/.*", line).groups()[0]
                if category in used_categories:
                    shutil.move(EXTRACTPATH/line, destdir/line)

    # The rest is for training
    destdir = EXTRACTPATH/"train"
    os.mkdir(destdir)
    for category in used_categories:
        shutil.move(str(EXTRACTPATH/category), destdir)


if __name__ == "__main__":
    os.makedirs(EXTRACTPATH, exist_ok=True)
    download_tarball()
    extract_tarball()
    organize_files()
