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

import unittest

import torch

from nupic.torch.modules import SparseWeights, SparseWeights2d


class TestSparseWeights(unittest.TestCase):

    def test_sparse_weights_1d(self):
        in_features, out_features = 784, 10
        with torch.no_grad():
            for percent_on in [0.1, 0.5, 0.9]:
                linear = torch.nn.Linear(in_features=in_features,
                                         out_features=out_features)
                sparse = SparseWeights(linear, percent_on)
                nonzeros = torch.nonzero(sparse.module.weight, as_tuple=True)[0]
                counts = torch.unique(nonzeros, return_counts=True)[1]

                # Expected non-zeros per output feature
                expected = [round(in_features * percent_on)] * out_features
                self.assertSequenceEqual(counts.numpy().tolist(), expected)

    def test_sparse_weights_2d(self):
        in_channels, kernel_size, out_channels = 64, (5, 5), 64
        input_size = in_channels * kernel_size[0] * kernel_size[1]

        with torch.no_grad():
            for percent_on in [0.1, 0.5, 0.9]:
                cnn = torch.nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size)
                sparse = SparseWeights2d(cnn, percent_on)
                nonzeros = torch.nonzero(sparse.module.weight, as_tuple=True)[0]
                counts = torch.unique(nonzeros, return_counts=True)[1]

                # Expected non-zeros per output channel
                expected = [round(input_size * percent_on)] * out_channels
                self.assertSequenceEqual(counts.numpy().tolist(), expected)


if __name__ == "__main__":
    unittest.main()
