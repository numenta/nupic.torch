# ------------------------------------------------------------------------------
#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
# ------------------------------------------------------------------------------
from os import path

from setuptools import find_namespace_packages, setup

import nupic.torch

ROOT = path.abspath(path.dirname(__file__))

# Get requirements from file
with open(path.join(ROOT, "requirements.txt")) as f:
    requirements = [line.strip() for line in f.readlines() if not line.startswith("#")]

# Get requirements-dev from file
with open(path.join(ROOT, "requirements-dev.txt")) as f:
    requirements_dev = [
        line.strip() for line in f.readlines() if not line.startswith("#")
    ]

# Get the long description from the README file
with open(path.join(ROOT, "README.md")) as f:
    readme = f.read()

setup(
    name="nupic.torch",
    author="Numenta",
    author_email="help@numenta.org",
    license="AGPLv3",
    platforms=["any"],
    url="https://github.com/numenta/nupic.torch",
    description="Numenta Platform for Intelligent Computing PyTorch libraries",
    long_description=readme,
    long_description_content_type="text/markdown",
    version=nupic.torch.__version__,
    packages=find_namespace_packages(include=["nupic.*"]),
    install_requires=requirements,
    python_requires=">=3.6, <4",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later "
        "(AGPLv3+)",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        'Documentation': "https://nupictorch.readthedocs.io",
        "Bug Reports": "https://github.com/numenta/nupic.torch/issues",
        "Source": "https://github.com/numenta/nupic.torch",
    },
    setup_requires=requirements_dev,
    test_suite="tests",
    tests_require=["pytest>=4.4.0"],
)
