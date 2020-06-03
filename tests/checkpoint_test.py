#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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
from torch.hub import load_state_dict_from_url

from nupic.torch.models.sparse_cnn import gsc_sparse_cnn

GSC_SPARSE_CNN_EAC5F79F = "http://public.numenta.com/pytorch/hub/gsc_sparse_cnn-eac5f79f.pth"  # noqa: E501


class SerializationTestCase(unittest.TestCase):

    def test_checkpoint_backward_compatibility(self):

        # Make sure current model is compatible with old checkpoint
        model1 = gsc_sparse_cnn(pretrained=True)
        model2 = gsc_sparse_cnn(pretrained=False)
        state_dict = load_state_dict_from_url(GSC_SPARSE_CNN_EAC5F79F)
        model2.load_state_dict(state_dict)
        model1.eval()
        model2.eval()
        x = torch.randn((16, 1, 32, 32))
        with torch.no_grad():
            y1 = model1(x).numpy()
            y2 = model2(x).numpy()
            self.assertAlmostEqual(y1.sum(), y2.sum())


if __name__ == "__main__":
    unittest.main()
