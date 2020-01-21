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
import abc
import math

import numpy as np
import torch
import torch.nn as nn


def rezero_weights(m):
    """Function used to update the weights after each epoch.

    Call using :meth:`torch.nn.Module.apply` after each epoch if required
    For example: ``m.apply(rezero_weights)``

    :param m: SparseWeightsBase module
    """
    if isinstance(m, SparseWeightsBase):
        if m.training:
            m.rezero_weights()


def normalize_sparse_weights(m):
    """Initialize the weights using kaiming_uniform initialization normalized
    to the number of non-zeros in the layer instead of the whole input size.

    Similar to torch.nn.Linear.reset_parameters() but applying weight
    sparsity to the input size
    """
    if isinstance(m, SparseWeightsBase):
        _, input_size = m.module.weight.shape
        fan = int(input_size * m.weight_sparsity)
        gain = nn.init.calculate_gain("leaky_relu", math.sqrt(5))
        std = gain / np.math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        nn.init.uniform_(m.module.weight, -bound, bound)
        if m.module.bias is not None:
            bound = 1 / math.sqrt(fan)
            nn.init.uniform_(m.module.bias, -bound, bound)


class SparseWeightsBase(nn.Module, metaclass=abc.ABCMeta):
    """
    Base class for the all Sparse Weights modules.

    :param module:
      The module to sparsify the weights
    :param weight_sparsity:
      Pct of weights that are allowed to be non-zero in the layer.
    """

    def __init__(self, module, weight_sparsity):
        super(SparseWeightsBase, self).__init__()
        assert 0 < weight_sparsity < 1

        self.module = module
        self.weight_sparsity = weight_sparsity
        self.register_buffer("zero_weights", self.compute_indices())
        self.rezero_weights()

    def extra_repr(self):
        return "weight_sparsity={}".format(self.weight_sparsity)

    def forward(self, x):
        if self.training:
            self.rezero_weights()
        return self.module(x)

    @abc.abstractmethod
    def compute_indices(self):
        """For each unit, decide which weights are going to be zero.

        :return: tensor indices for all non-zero weights. See :meth:`rezeroWeights`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rezero_weights(self):
        """Set the previously selected weights to zero.

        See :meth:`computeIndices`
        """
        raise NotImplementedError


class SparseWeights(SparseWeightsBase):
    """Enforce weight sparsity on linear module during training.

    Sample usage:

      model = nn.Linear(784, 10)
      model = SparseWeights(model, 0.4)

    :param module:
      The module to sparsify the weights
    :param weight_sparsity:
      Pct of weights that are allowed to be non-zero in the layer.
    """

    def __init__(self, module, weight_sparsity):
        super(SparseWeights, self).__init__(module, weight_sparsity)
        assert isinstance(module, nn.Linear)

    def compute_indices(self):
        # For each unit, decide which weights are going to be zero
        output_size, input_size = self.module.weight.shape
        num_zeros = int(round((1.0 - self.weight_sparsity) * input_size))

        output_indices = np.arange(output_size)
        input_indices = np.array(
            [np.random.permutation(input_size)[:num_zeros] for _ in output_indices],
            dtype=np.long,
        )

        # Create tensor indices for all non-zero weights
        zero_indices = np.empty((output_size, num_zeros, 2), dtype=np.long)
        zero_indices[:, :, 0] = output_indices[:, None]
        zero_indices[:, :, 1] = input_indices
        zero_indices = zero_indices.reshape(-1, 2)
        return torch.from_numpy(zero_indices.transpose())

    def rezero_weights(self):
        zero_idx = (self.zero_weights[0].long(), self.zero_weights[1].long())
        self.module.weight.data[zero_idx] = 0.0


class SparseWeights2d(SparseWeightsBase):
    """Enforce weight sparsity on CNN modules Sample usage:

      model = nn.Conv2d(in_channels, out_channels, kernel_size, ...)
      model = SparseWeights2d(model, 0.4)

    :param module:
      The module to sparsify the weights
    :param weight_sparsity:
      Pct of weights that are allowed to be non-zero in the layer.
    """

    def __init__(self, module, weight_sparsity):
        super(SparseWeights2d, self).__init__(module, weight_sparsity)
        assert isinstance(module, nn.Conv2d)

    def compute_indices(self):
        # For each unit, decide which weights are going to be zero
        in_channels = self.module.in_channels
        out_channels = self.module.out_channels
        kernel_size = self.module.kernel_size

        input_size = in_channels * kernel_size[0] * kernel_size[1]
        num_zeros = int(round((1.0 - self.weight_sparsity) * input_size))

        output_indices = np.arange(out_channels)
        input_indices = np.array(
            [np.random.permutation(input_size)[:num_zeros] for _ in output_indices],
            dtype=np.long,
        )

        # Create tensor indices for all non-zero weights
        zero_indices = np.empty((out_channels, num_zeros, 2), dtype=np.long)
        zero_indices[:, :, 0] = output_indices[:, None]
        zero_indices[:, :, 1] = input_indices
        zero_indices = zero_indices.reshape(-1, 2)

        return torch.from_numpy(zero_indices.transpose())

    def rezero_weights(self):
        zero_idx = (self.zero_weights[0].long(), self.zero_weights[1].long())
        self.module.weight.data.view(self.module.out_channels, -1)[zero_idx] = 0.0
