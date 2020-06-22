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

from .sparse_weights import SparseWeights, SparseWeights2d


class PrunableSparseWeightBase(object):
    """
    Enable easy setting and getting of the off-mask that defines which
    weights are zero.
    """

    @property
    def off_mask(self):
        """
        Gets the value of `zero_mask` in bool format. Thus one may call
        ```
        self.weight[~self.off_mask]  # returns weights that are currently on
        ```
        """
        return self.zero_mask.bool()

    @off_mask.setter
    def off_mask(self, mask):
        """
        Sets the values of `zero_mask`, updating self.sparsity to reflect the
        sparsity of the new mask.
        """
        self.sparsity = mask.sum().item() / mask.numel()
        self.zero_mask[:] = mask


class PrunableSparseWeights(SparseWeights, PrunableSparseWeightBase):
    """
    Enforce weight sparsity on linear module. The off-weights may be
    changed dynamically through the `off_mask` property.
    """
    def __init__(self, module, weight_sparsity=None, sparsity=None):
        super().__init__(
            module, weight_sparsity=weight_sparsity, sparsity=sparsity,
            allow_extremes=True
        )


class PrunableSparseWeights2d(SparseWeights2d, PrunableSparseWeightBase):
    """
    Enforce weight sparsity on CNN modules. The off-weights may be
    changed dynamically through the `off_mask` property.
    """

    def __init__(self, module, weight_sparsity=None, sparsity=None):
        super().__init__(
            module, weight_sparsity=weight_sparsity, sparsity=sparsity,
            allow_extremes=True
        )
