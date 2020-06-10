# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

import torch
import torch.nn as nn

from .sparse_weights import SparseWeights, SparseWeights2d


class PrunableSparseWeightBase(object):
    """
    Enable easy setting and getting of the off-mask that defines which
    weights are zero.
    """

    @property
    def off_mask(self):
        """
        Accesses a boolean mask of size `self.weight.shape` which have non-zero
        entries as defined by `zero_weights`. Thus one may call
        ```
        self.weight[~self.off_mask]  # returns weights that are currently on
        ```
        """
        out_shape = self.module.weight.shape[0]
        zero_idx = (self.zero_weights[0].long(), self.zero_weights[1].long())
        weight_mask = torch.zeros_like(self.module.weight).bool()
        weight_mask.view(out_shape, -1)[zero_idx] = 1
        return weight_mask

    @off_mask.setter
    def off_mask(self, mask):
        """
        Sets the values of `zero_weights` according to a mask of size
        `self.weight.shape`.
        """
        mask = mask.bool()
        self.sparsity = mask.sum().item() / mask.numel()
        out_shape = self.module.weight.shape[0]
        self.zero_weights = mask.view(out_shape, -1).nonzero().permute(1, 0)


class PrunableSparseWeights(SparseWeights, PrunableSparseWeightBase):
    """
    Enforce weight sparsity on linear module. The off-weights may be
    changed dynamically through the `off_mask` property.
    """
    def __init__(self, module, weight_sparsity=None, sparsity=None):
        assert isinstance(module, nn.Linear)
        assert 0 <= (weight_sparsity or sparsity) <= 1
        super(SparseWeights, self).__init__(
            module, weight_sparsity=weight_sparsity, sparsity=sparsity
        )


class PrunableSparseWeights2d(SparseWeights2d, PrunableSparseWeightBase):
    """
    Enforce weight sparsity on CNN modules. The off-weights may be
    changed dynamically through the `off_mask` property.
    """

    def __init__(self, module, weight_sparsity=None, sparsity=None):
        assert isinstance(module, nn.Conv2d)
        assert 0 <= (weight_sparsity or sparsity) <= 1
        super(SparseWeights2d, self).__init__(
            module, weight_sparsity=weight_sparsity, sparsity=sparsity
        )
