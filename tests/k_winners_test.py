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

import nupic.torch.functions as F
from nupic.torch.modules import KWinners


class TestContext(object):
    def __init__(self):
        self.saved_tensors = None

    def save_for_backward(self, x):
        self.saved_tensors = (x,)


class KWinnersTest(unittest.TestCase):
    """"""

    def setUp(self):

        # Batch size 2
        x = torch.ones((2, 7))
        x[0, 1] = 1.2
        x[0, 2] = 1.1
        x[0, 3] = 1.3
        x[0, 5] = 1.5
        x[1, 0] = 1.1
        x[1, 2] = 1.2
        x[1, 4] = 1.3
        x[1, 6] = 1.2
        self.x = x
        self.gradient = torch.rand(x.shape)

        # All equal
        self.duty_cycle = torch.zeros((2, 7))
        self.duty_cycle[:] = 1.0 / 3.0

        # Batch size 2
        x2 = torch.ones((2, 6))
        x2[0, 0] = 1.5
        x2[0, 2] = 1.1
        x2[0, 3] = 1.3
        x2[1, 0] = 1.1
        x2[1, 2] = 1.2
        x2[1, 3] = 1.6
        self.x2 = x2

    def test_one(self):
        """Equal duty cycle, boost factor 0, k=3, batch size 2."""
        x = self.x

        ctx = TestContext()

        result = F.KWinners.forward(ctx, x, self.duty_cycle, k=3, boost_strength=0.0)

        expected = torch.zeros_like(x)
        expected[0, 1] = 1.2
        expected[0, 3] = 1.3
        expected[0, 5] = 1.5
        expected[1, 2] = 1.2
        expected[1, 4] = 1.3
        expected[1, 6] = 1.2

        self.assertEqual(result.shape, expected.shape)

        num_correct = (result == expected).sum()
        self.assertEqual(num_correct, result.numel())

        # Test that mask saved by forward has 1s in the right places
        mask = ctx.saved_tensors[0]
        expected_mask = torch.FloatTensor([
            [0, 1, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0, 1]])
        self.assertEqual((mask == expected_mask).sum(), mask.numel())

        # Test that grad_x returned by backwards is equal to the masked
        # gradients before backward.
        grad_x, _, _, _ = F.KWinners.backward(ctx, self.gradient)

        self.assertAlmostEqual(
            grad_x.sum().item(),
            (mask * self.gradient).sum().item(),
            places=4,
        )
        self.assertEqual(len(grad_x.nonzero()), len(expected.nonzero()))

    def test_k_winners_module(self):
        x = self.x2

        n = 6

        kw = KWinners(
            n,
            percent_on=0.333,
            boost_strength=1.0,
            boost_strength_factor=0.5,
            duty_cycle_period=1000,
        )

        kw.train()  # Testing with mod.training = True

        # Expect 2 winners per batch (33% of 6)
        expected = torch.zeros_like(x)
        expected[0, 0] = 1.5
        expected[0, 3] = 1.3
        expected[1, 2] = 1.2
        expected[1, 3] = 1.6
        result = kw(x)
        self.assertEqual(result.shape, expected.shape)

        num_correct = (result == expected).sum()
        self.assertEqual(num_correct, result.reshape(-1).size()[0])

        new_duty = torch.tensor([1.0, 0, 1.0, 2.000, 0, 0]) / 2.0

        diff = (kw.duty_cycle - new_duty).abs().sum()
        self.assertLessEqual(diff, 0.001)


if __name__ == "__main__":
    unittest.main()
