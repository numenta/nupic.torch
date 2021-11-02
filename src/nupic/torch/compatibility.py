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

from collections import OrderedDict

import torch


def upgrade_to_masked_sparseweights(state_dict):
    """
    Returns a new state dict with any "zero_weights" tensors converted to
    "zero_mask" tensors. (The "zero_weights" was a list of indices of zeroes in
    the weight tensor.)
    """
    upgraded = []
    for name, tensor in state_dict.items():
        if "zero_weights" in name:
            weight_name = name.replace("zero_weights", "module.weight")
            zero_mask = torch.zeros(state_dict[weight_name].shape,
                                    device=tensor.device)
            if tensor.shape[0] == 2:
                # Assume this is the standard previous format of SparseWeights
                # and SparseWeights2d
                zero_mask.view(zero_mask.shape[0], -1)[tuple(tensor)] = 1
            else:
                # Assume the tensor is a valid index list for the weight shape
                zero_mask[tuple(tensor)] = 1

            upgraded.append((name.replace("zero_weights", "zero_mask"),
                             zero_mask))
        else:
            upgraded.append((name, tensor))

    return OrderedDict(upgraded)
