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

import unittest

import torch

from nupic.torch.duty_cycle_metrics import binary_entropy, max_entropy


class DutyCycleMetricsTest(unittest.TestCase):
    """Simplistic tests of duty cycle entropy metrics."""

    def test_binary_entropy(self):

        p = torch.tensor([0.1, 0.02, 0.99, 0.5, 0.75, 0.8, 0.3, 0.4, 0.0, 1.0])
        entropy, entropy_sum = binary_entropy(p)
        self.assertAlmostEqual(entropy_sum.item(), 5.076676985, places=4)
        self.assertAlmostEqual(entropy_sum.item(), entropy.sum(), places=4)
        self.assertAlmostEqual(entropy[0].item(), 0.468995594, places=4)
        self.assertAlmostEqual(entropy[1].item(), 0.141440543, places=4)
        self.assertAlmostEqual(entropy[2].item(), 0.080793136, places=4)
        self.assertEqual(entropy[8].item(), 0.0)
        self.assertEqual(entropy[9].item(), 0.0)

        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        entropy, entropy_sum = binary_entropy(p)
        self.assertAlmostEqual(entropy_sum, 3.245112498, places=4)
        self.assertAlmostEqual(entropy_sum, entropy.sum(), places=4)

        p = torch.tensor([0.5, 0.5, 0.5, 0.5])
        entropy, entropy_sum = binary_entropy(p)
        self.assertAlmostEqual(entropy_sum, 4.0, places=4)
        self.assertAlmostEqual(entropy_sum, entropy.sum(), places=4)
        self.assertAlmostEqual(entropy[0], 1.0, places=4)
        self.assertAlmostEqual(entropy[1], 1.0, places=4)
        self.assertAlmostEqual(entropy[2], 1.0, places=4)
        self.assertAlmostEqual(entropy[3], 1.0, places=4)

    def test_max_entropy(self):

        entropy = max_entropy(1, 1)
        self.assertAlmostEqual(entropy, 0.0, places=4)

        entropy = max_entropy(1, 0)
        self.assertAlmostEqual(entropy, 0.0, places=4)

        entropy = max_entropy(4, 1)
        self.assertAlmostEqual(entropy, 3.245112498, places=4)

        entropy = max_entropy(4, 2)
        self.assertAlmostEqual(entropy, 4.0, places=4)

        entropy = max_entropy(100, 1)
        self.assertAlmostEqual(entropy, 8.07931359, places=4)

        entropy = max_entropy(100, 10)
        self.assertAlmostEqual(entropy, 46.89955936, places=4)

        entropy = max_entropy(2048, 40)
        self.assertAlmostEqual(entropy, 284.2634199, places=4)


if __name__ == "__main__":
    unittest.main()
