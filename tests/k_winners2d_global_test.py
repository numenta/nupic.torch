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
from nupic.torch.modules import KWinners2d


class TestContext(object):
    def __init__(self):
        self.saved_tensors = None

    def save_for_backward(self, x):
        self.saved_tensors = (x,)


class KWinners2DTest(unittest.TestCase):
    """"""

    def setUp(self):
        # Tests will use 3 filters and image width, height = 2 X 2

        # Batch size 1
        x = torch.rand(1, 3, 2, 2) / 2.0
        x[0, 0, 1, 0] = 1.10
        x[0, 0, 1, 1] = 1.20
        x[0, 1, 0, 1] = 1.21
        x[0, 2, 1, 0] = 1.30
        self.x = x
        self.gradient = torch.rand(x.shape)

        # Batch size 2
        x = torch.rand(2, 3, 2, 2) / 2.0
        x[0, 0, 1, 0] = 1.10
        x[0, 0, 1, 1] = 1.20
        x[0, 1, 0, 1] = 1.21
        x[0, 2, 1, 0] = 1.30

        x[1, 0, 0, 0] = 1.40
        x[1, 1, 0, 0] = 1.50
        x[1, 1, 0, 1] = 1.60
        x[1, 2, 1, 1] = 1.70
        self.x2 = x
        self.gradient2 = torch.rand(x.shape)

        # All equal
        self.duty_cycle = torch.zeros((1, 3, 1, 1))
        self.duty_cycle[:] = 1.0 / 3.0

    def test_one(self):
        """Equal duty cycle, boost factor 0, k=4, batch size 1."""
        x = self.x

        ctx = TestContext()

        result = F.KWinners2dGlobal.forward(ctx, x, self.duty_cycle, k=4,
                                            boost_strength=0.0)

        expected = torch.zeros_like(x)
        expected[0, 0, 1, 0] = x[0, 0, 1, 0]
        expected[0, 0, 1, 1] = x[0, 0, 1, 1]
        expected[0, 1, 0, 1] = x[0, 1, 0, 1]
        expected[0, 2, 1, 0] = x[0, 2, 1, 0]

        self.assertEqual(result.shape, expected.shape)

        num_correct = (result == expected).sum()
        self.assertEqual(num_correct, result.reshape(-1).size()[0])

        indices = ctx.saved_tensors[0].reshape(-1).sort()[0]
        expected_indices = torch.tensor([2, 3, 5, 10])
        num_correct = (indices == expected_indices).sum()
        self.assertEqual(num_correct, 4)

        # Test that gradient values are in the right places, that their sum is
        # equal, and that they have exactly the right number of nonzeros
        grad_x, _, _, _ = F.KWinners2dGlobal.backward(ctx, self.gradient)
        grad_x = grad_x.reshape(-1)
        self.assertEqual(
            (grad_x[indices] == self.gradient.reshape(-1)[indices]).sum(), 4
        )
        self.assertAlmostEqual(
            grad_x.sum().item(),
            self.gradient.reshape(-1)[indices].sum().item(),
            places=4,
        )
        self.assertEqual(len(grad_x.nonzero()), 4)

    def test_two(self):
        """Equal duty cycle, boost factor 0, k=3."""
        x = self.x

        ctx = TestContext()

        result = F.KWinners2dGlobal.forward(ctx, x, self.duty_cycle, k=3,
                                            boost_strength=0.0)

        expected = torch.zeros_like(x)
        expected[0, 0, 1, 1] = x[0, 0, 1, 1]
        expected[0, 1, 0, 1] = x[0, 1, 0, 1]
        expected[0, 2, 1, 0] = x[0, 2, 1, 0]

        self.assertEqual(result.shape, expected.shape)

        num_correct = (result == expected).sum()
        self.assertEqual(num_correct, result.reshape(-1).size()[0])

        indices = ctx.saved_tensors[0].reshape(-1).sort()[0]
        expected_indices = torch.tensor([3, 5, 10])
        num_correct = (indices == expected_indices).sum()
        self.assertEqual(num_correct, 3)

        # Test that gradient values are in the right places, that their sum is
        # equal, and that they have exactly the right number of nonzeros
        grad_x, _, _, _ = F.KWinners2dGlobal.backward(ctx, self.gradient)
        grad_x = grad_x.reshape(-1)
        self.assertEqual(
            (grad_x[indices] == self.gradient.reshape(-1)[indices]).sum(), 3
        )
        self.assertAlmostEqual(
            grad_x.sum().item(),
            self.gradient.reshape(-1)[indices].sum().item(),
            places=4,
        )
        self.assertEqual(len(grad_x.nonzero()), 3)

    def test_three(self):
        """Equal duty cycle, boost factor=0, k=4, batch size=2."""
        x = self.x2

        ctx = TestContext()

        result = F.KWinners2dGlobal.forward(ctx, x, self.duty_cycle, k=4,
                                            boost_strength=0.0)

        expected = torch.zeros_like(x)
        expected[0, 0, 1, 0] = x[0, 0, 1, 0]
        expected[0, 0, 1, 1] = x[0, 0, 1, 1]
        expected[0, 1, 0, 1] = x[0, 1, 0, 1]
        expected[0, 2, 1, 0] = x[0, 2, 1, 0]
        expected[1, 0, 0, 0] = x[1, 0, 0, 0]
        expected[1, 1, 0, 0] = x[1, 1, 0, 0]
        expected[1, 1, 0, 1] = x[1, 1, 0, 1]
        expected[1, 2, 1, 1] = x[1, 2, 1, 1]

        self.assertEqual(result.shape, expected.shape)

        num_correct = (result == expected).sum()
        self.assertEqual(num_correct, result.reshape(-1).size()[0])

        indices = ctx.saved_tensors[0].sort()[0]
        expected_indices = torch.tensor([[2, 3, 5, 10], [0, 4, 5, 11]])
        num_correct = (indices == expected_indices).sum()
        self.assertEqual(num_correct, 8)

        # Test that gradient values are in the right places, that their sum is
        # equal, and that they have exactly the right number of nonzeros
        out_grad, _, _, _ = F.KWinners2dGlobal.backward(ctx, self.gradient2)
        out_grad = out_grad.reshape(2, -1)
        in_grad = self.gradient2.reshape(2, -1)
        self.assertEqual((out_grad == in_grad).sum(), 8)
        self.assertEqual(len(out_grad.nonzero()), 8)

    def test_four(self):
        """Equal duty cycle, boost factor=0, k=3, batch size=2."""
        x = self.x2

        ctx = TestContext()

        result = F.KWinners2dGlobal.forward(ctx, x, self.duty_cycle, k=3,
                                            boost_strength=0.0)

        expected = torch.zeros_like(x)
        expected[0, 0, 1, 1] = x[0, 0, 1, 1]
        expected[0, 1, 0, 1] = x[0, 1, 0, 1]
        expected[0, 2, 1, 0] = x[0, 2, 1, 0]
        expected[1, 1, 0, 0] = x[1, 1, 0, 0]
        expected[1, 1, 0, 1] = x[1, 1, 0, 1]
        expected[1, 2, 1, 1] = x[1, 2, 1, 1]

        self.assertEqual(result.shape, expected.shape)

        num_correct = (result == expected).sum()
        self.assertEqual(num_correct, result.reshape(-1).size()[0])

        indices = ctx.saved_tensors[0].sort()[0]
        expected_indices = torch.tensor([[3, 5, 10], [4, 5, 11]])
        num_correct = (indices == expected_indices).sum()
        self.assertEqual(num_correct, 6)

        # Test that gradient values are in the right places, that their sum is
        # equal, and that they have exactly the right number of nonzeros
        out_grad, _, _, _ = F.KWinners2dGlobal.backward(ctx, self.gradient2)
        out_grad = out_grad.reshape(2, -1)
        in_grad = self.gradient2.reshape(2, -1)
        self.assertEqual((out_grad == in_grad).sum(), 6)
        self.assertEqual(len(out_grad.nonzero()), 6)

    def test_k_winners2d_module_one(self):
        x = self.x2

        kw = KWinners2d(
            percent_on=0.333,
            channels=3,
            k_inference_factor=0.5,
            boost_strength=1.0,
            boost_strength_factor=0.5,
            duty_cycle_period=1000,
            local=False
        )

        expected = torch.zeros_like(x)
        expected[0, 0, 1, 0] = x[0, 0, 1, 0]
        expected[0, 0, 1, 1] = x[0, 0, 1, 1]
        expected[0, 1, 0, 1] = x[0, 1, 0, 1]
        expected[0, 2, 1, 0] = x[0, 2, 1, 0]
        expected[1, 0, 0, 0] = x[1, 0, 0, 0]
        expected[1, 1, 0, 0] = x[1, 1, 0, 0]
        expected[1, 1, 0, 1] = x[1, 1, 0, 1]
        expected[1, 2, 1, 1] = x[1, 2, 1, 1]

        result = kw(x)
        self.assertEqual(result.shape, expected.shape)

        num_correct = (result == expected).sum()
        self.assertEqual(num_correct, result.reshape(-1).size()[0])

        new_duty = torch.tensor([1.5000, 1.5000, 1.0000]) / 4.0
        diff = (kw.duty_cycle.reshape(-1) - new_duty).abs().sum()
        self.assertLessEqual(diff, 0.001)

    def test_k_winners2d_module_two(self):
        """
        Test a series of calls on the module in training mode.
        """

        x = self.x2

        expected = torch.zeros_like(x)
        expected[0, 0, 1, 0] = x[0, 0, 1, 0]
        expected[0, 0, 1, 1] = x[0, 0, 1, 1]
        expected[0, 2, 1, 0] = x[0, 2, 1, 0]
        expected[1, 0, 0, 0] = x[1, 0, 0, 0]
        expected[1, 1, 0, 1] = x[1, 1, 0, 1]
        expected[1, 2, 1, 1] = x[1, 2, 1, 1]

        kw = KWinners2d(
            percent_on=0.25,
            channels=3,
            k_inference_factor=0.5,
            boost_strength=1.0,
            boost_strength_factor=0.5,
            duty_cycle_period=1000,
            local=False
        )

        kw.train(mode=True)
        result = kw(x)
        result = kw(x)
        result = kw(x)
        result = kw(x)
        result = kw(x)
        result = kw(x)

        self.assertTrue(result.eq(expected).all())


if __name__ == "__main__":
    unittest.main()
