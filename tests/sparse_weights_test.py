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

from nupic.torch.modules import (
    PrunableSparseWeights,
    PrunableSparseWeights2d,
    SparseWeights,
    SparseWeights2d,
)


class TestSparseWeights(unittest.TestCase):

    def test_sparse_weights_1d(self):
        in_features, out_features = 784, 10
        with torch.no_grad():
            for sparsity in [0.1, 0.5, 0.9]:
                linear = torch.nn.Linear(in_features=in_features,
                                         out_features=out_features)
                sparse = SparseWeights(linear, sparsity=sparsity)
                nonzeros = torch.nonzero(sparse.module.weight, as_tuple=True)[0]
                counts = torch.unique(nonzeros, return_counts=True)[1]

                # Expected non-zeros per output feature
                expected = [round(in_features * (1.0 - sparsity))] * out_features
                self.assertSequenceEqual(counts.numpy().tolist(), expected)

    def test_sparse_weights_2d(self):
        in_channels, kernel_size, out_channels = 64, (5, 5), 64
        input_size = in_channels * kernel_size[0] * kernel_size[1]

        with torch.no_grad():
            for sparsity in [0.1, 0.5, 0.9]:
                cnn = torch.nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size)
                sparse = SparseWeights2d(cnn, sparsity=sparsity)
                nonzeros = torch.nonzero(sparse.module.weight, as_tuple=True)[0]
                counts = torch.unique(nonzeros, return_counts=True)[1]

                # Expected non-zeros per output channel
                expected = [round(input_size * (1.0 - sparsity))] * out_channels
                self.assertSequenceEqual(counts.numpy().tolist(), expected)

    def test_rezero_after_forward_1d(self):
        in_features, out_features = 784, 10
        for sparsity in [0.1, 0.5, 0.9]:
            linear = torch.nn.Linear(in_features=in_features,
                                     out_features=out_features)
            sparse = SparseWeights(linear, sparsity=sparsity)

            # Ensure weights are not sparse
            sparse.module.weight.data.fill_(1.0)
            sparse.train()
            x = torch.ones((1,) + (in_features,))
            sparse(x)

            # When training, the forward function should set weights back to zero.
            nonzeros = torch.nonzero(sparse.module.weight, as_tuple=True)[0]
            counts = torch.unique(nonzeros, return_counts=True)[1]
            expected = [round(in_features * (1.0 - sparsity))] * out_features
            self.assertSequenceEqual(counts.numpy().tolist(), expected)

    def test_rezero_after_forward_2d(self):
        in_channels, kernel_size, out_channels = 64, (5, 5), 64
        input_size = in_channels * kernel_size[0] * kernel_size[1]

        with torch.no_grad():
            for sparsity in [0.1, 0.5, 0.9]:
                cnn = torch.nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size)
                sparse = SparseWeights2d(cnn, sparsity=sparsity)

                # Ensure weights are not sparse
                sparse.module.weight.data.fill_(1.0)
                sparse.train()
                x = torch.ones((1,) + (in_channels, kernel_size[0], kernel_size[1]))
                sparse(x)

                # When training, the forward function should set weights back to zero.
                nonzeros = torch.nonzero(sparse.module.weight, as_tuple=True)[0]
                counts = torch.unique(nonzeros, return_counts=True)[1]
                expected = [round(input_size * (1.0 - sparsity))] * out_channels
                self.assertSequenceEqual(counts.numpy().tolist(), expected)

    def test_prunable_dense_linear(self):

        lin = torch.nn.Linear(10, 10)
        sw = PrunableSparseWeights(lin, sparsity=0)

        self.assertTrue(torch.all(sw.off_mask == torch.zeros_like(sw.weight)))
        sw.weight[:] = 1
        sw.rezero_weights()
        self.assertTrue(sw.weight.sum() == sw.weight.numel())

    def test_prunable_sparse_linear(self):

        lin = torch.nn.Linear(10, 10)
        sw = PrunableSparseWeights(lin, sparsity=1)

        self.assertTrue(torch.all(sw.off_mask == torch.ones_like(sw.weight)))
        sw.weight[:] = 1
        sw.rezero_weights()
        self.assertTrue(sw.weight.sum() == 0)

    def test_prunable_dense_conv(self):

        conv = torch.nn.Conv2d(4, 4, 3)
        sw = PrunableSparseWeights2d(conv, sparsity=0)

        self.assertTrue(torch.all(sw.off_mask == torch.zeros_like(sw.weight)))
        sw.weight[:] = 1
        sw.rezero_weights()
        self.assertTrue(sw.weight.sum() == sw.weight.numel())

    def test_prunable_sparse_conv(self):

        conv = torch.nn.Conv2d(4, 4, 3)
        sw = PrunableSparseWeights2d(conv, sparsity=1)

        self.assertTrue(torch.all(sw.off_mask == torch.ones_like(sw.weight)))
        sw.weight[:] = 1
        sw.rezero_weights()
        self.assertTrue(sw.weight.sum() == 0)

    def test_linear_prunable_off_mask(self):

        lin = torch.nn.Linear(4, 4)
        sw = PrunableSparseWeights(lin, sparsity=0)

        sw.off_mask = torch.tensor([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 0, 0]
        ])
        self.assertTrue(sw.sparsity == 6 / 16)
        sw.weight[:] = 1
        sw.rezero_weights()
        self.assertTrue(sw.weight.sum() == 10)

    def test_conv_prunable_off_mask(self):

        conv = torch.nn.Conv2d(2, 2, 2, 2)
        sw = PrunableSparseWeights2d(conv, sparsity=0)

        sw.off_mask = torch.tensor([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 0, 0]
        ]).view(2, 2, 2, 2)
        self.assertTrue(sw.sparsity == 6 / 16)
        sw.weight[:] = 1
        sw.rezero_weights()
        self.assertTrue(sw.weight.sum() == 10)


if __name__ == "__main__":
    unittest.main()
