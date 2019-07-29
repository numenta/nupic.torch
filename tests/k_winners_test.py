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

        # All equal duty cycle for x.
        self.duty_cycle = torch.zeros(7)
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

        # Unequal duty cycle for x2.
        duty_cycle2 = torch.zeros(6)
        duty_cycle2[0] = 1.0 / 2.0
        duty_cycle2[1] = 1.0 / 4.0
        duty_cycle2[2] = 1.0 / 2.0
        duty_cycle2[3] = 1.0 / 4.0
        duty_cycle2[4] = 1.0 / 2.0
        duty_cycle2[5] = 1.0 / 4.0
        self.duty_cycle2 = duty_cycle2

        # Batch size 2, but with negative numbers.
        x3 = torch.ones((2, 6))
        x3[0, 1] = -1.2
        x3[0, 2] = 1.2
        x3[1, 1] = 1.2
        x3[1, 2] = -1.2
        self.x3 = x3

        # Unequal duty cycle for x3.
        duty_cycle3 = torch.zeros(6)
        duty_cycle3[1] = 0.001
        duty_cycle3[2] = 100
        self.duty_cycle3 = duty_cycle3

        # Batch size 1.
        x4 = torch.ones((1, 10))
        x4[0, 2] = 1.2
        x4[0, 3] = 1.2
        x4[0, 4] = 1.2
        x4[0, 5] = 1.2
        x4[0, 6] = 1.3
        x4[0, 7] = 1.3
        x4[0, 8] = 1.3
        x4[0, 9] = 1.3
        self.x4 = x4

        # All equal duty cycle for x.
        self.duty_cycle4 = torch.zeros(10)
        self.duty_cycle4[:] = 1.0 / 10.0

    def test_one(self):
        """
        Equal duty cycle, boost factor 0 (and then over a range), k=3, batch size 2.
        """

        # Set up test input and context.
        x = self.x
        ctx = TestContext()

        # Test forward with boost factor of 0.
        result = F.KWinners.forward(ctx, x, self.duty_cycle, k=3, boost_strength=0.0)

        expected = torch.zeros_like(x)
        expected[0, 1] = 1.2
        expected[0, 3] = 1.3
        expected[0, 5] = 1.5
        expected[1, 2] = 1.2
        expected[1, 4] = 1.3
        expected[1, 6] = 1.2

        self.assertEqual(result.shape, expected.shape)
        self.assertTrue(result.eq(expected).all())

        # Test that mask saved by forward has 1s in the right places
        mask = ctx.saved_tensors[0]
        expected_mask = torch.FloatTensor([
            [0, 1, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0, 1]])
        self.assertTrue(mask.eq(expected_mask).all())

        # Test that grad_x returned by backwards is equal to the masked
        # gradients before backward.
        grad_x, _, _, _ = F.KWinners.backward(ctx, self.gradient)
        self.assertTrue(grad_x.allclose(mask * self.gradient, rtol=0))

        # Test forward again with boost factor from 1 to 10. Should give save result
        # with an all equal duty cycle.
        for b in range(1, 10):

            result = F.KWinners.forward(ctx, x, self.duty_cycle, k=3, boost_strength=b)

            self.assertEqual(result.shape, expected.shape)
            self.assertTrue(result.eq(expected).all())

    def test_two(self):
        """
        Unequal duty cycle, boost factor 0 (and then over a range), k = 3, batch size 2.
        """

        # Set up test input and context.
        x = self.x2
        ctx = TestContext()

        # Test forward with boost factor of 0.
        result = F.KWinners.forward(ctx, x, self.duty_cycle2, k=3, boost_strength=0.0)

        expected = torch.zeros_like(x)
        expected[0, 0] = 1.5
        expected[0, 2] = 1.1
        expected[0, 3] = 1.3
        expected[1, 0] = 1.1
        expected[1, 2] = 1.2
        expected[1, 3] = 1.6

        self.assertEqual(result.shape, expected.shape)
        self.assertTrue(result.eq(expected).all())

        # Test forward again with boost factor of 1.
        result = F.KWinners.forward(ctx, x, self.duty_cycle2, k=3, boost_strength=1.0)

        expected = torch.zeros_like(x)
        expected[0, 0] = 1.5
        expected[0, 1] = 1.0
        expected[0, 3] = 1.3
        expected[1, 1] = 1.0
        expected[1, 3] = 1.6
        expected[1, 5] = 1.0

        self.assertEqual(result.shape, expected.shape)
        self.assertTrue(result.eq(expected).all())

        # Test forward again with boost factor from 2 to 10. Should give save result
        # given the differing duty cycles.
        expected = torch.zeros_like(x)
        expected[0, 1] = 1.0
        expected[0, 3] = 1.3
        expected[0, 5] = 1.0
        expected[1, 1] = 1.0
        expected[1, 3] = 1.6
        expected[1, 5] = 1.0

        for b in range(2, 10):

            result = F.KWinners.forward(ctx, x, self.duty_cycle2, k=3, boost_strength=b)

            self.assertEqual(result.shape, expected.shape)
            self.assertTrue(result.eq(expected).all())

    def test_three(self):
        """
        Unequal duty cycle, boost factor 0 (and then over a range), k = 3, batch size 2.
        """

        # Set up test input and context.
        x = self.x3
        ctx = TestContext()

        # Test forward with boost factor of 0.
        result = F.KWinners.forward(ctx, x, self.duty_cycle3, k=2, boost_strength=0.0)

        expected = torch.zeros_like(x)
        expected[0, 2] = 1.2
        expected[0, 3] = 1.0
        expected[1, 1] = 1.2
        expected[1, 5] = 1.0

        self.assertEqual(result.shape, expected.shape)
        self.assertTrue(result.eq(expected).all())

        # Test forward again with boost factor from 1 to 10. Should yield the same
        # result as the negative numbers will never be in the top k and the non-one
        # values have very large duty cycles.
        expected = torch.zeros_like(x)
        expected[0, 3] = 1.0
        expected[0, 5] = 1.0
        expected[1, 1] = 1.2
        expected[1, 5] = 1.0

        for b in range(1, 10):

            result = F.KWinners.forward(ctx, x, self.duty_cycle3, k=2, boost_strength=b)

            self.assertEqual(result.shape, expected.shape)
            self.assertTrue(result.eq(expected).all())

    def test_four(self):
        """
        All equal duty cycle, boost factor 0, k = 0,1, and n, batch size 1.
        """

        # Set up test input and context.
        x = self.x4
        ctx = TestContext()

        # Test forward with boost factor of 1 and k=0.
        result = F.KWinners.forward(ctx, x, self.duty_cycle4, k=0, boost_strength=1)

        expected = torch.zeros_like(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertTrue(result.eq(expected).all())

        # Test forward with boost factor of 1 and k=1.
        result = F.KWinners.forward(ctx, x, self.duty_cycle4, k=1, boost_strength=1)

        expected = torch.zeros_like(x)
        expected[0, -1] = 1.3

        self.assertEqual(result.shape, expected.shape)
        self.assertTrue(result.eq(expected).all())

        # Test forward with boost factor of 1 and k=1.
        result = F.KWinners.forward(ctx, x, self.duty_cycle4, k=10, boost_strength=1)

        expected = x.clone().detach()

        self.assertEqual(result.shape, expected.shape)
        self.assertTrue(result.eq(expected).all())

    def test_k_winners_module_one(self):

        # Set up test input and module.
        x = self.x2
        n = 6

        kw = KWinners(
            n,
            percent_on=0.333,
            k_inference_factor=1.5,
            boost_strength=1.0,
            boost_strength_factor=0.5,
            duty_cycle_period=1000,
        )

        # Test with mod.training = False.
        kw.train(mode=False)

        # Expect 3 winners per batch (1.5 * 33% of 6 is 1 / 2 of 6)
        expected = torch.zeros_like(x)
        expected[0, 0] = 1.5
        expected[0, 2] = 1.1
        expected[0, 3] = 1.3
        expected[1, 0] = 1.1
        expected[1, 2] = 1.2
        expected[1, 3] = 1.6
        result = kw(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertTrue(result.eq(expected).all())

        # Run forward pass again while still not in training mode.
        # Should give the same result as the duty cycles are not updated.
        result = kw(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertTrue(result.eq(expected).all())

        # Test with mod.training = True
        kw.train(mode=True)

        # Expect 2 winners per batch (33% of 6)
        expected = torch.zeros_like(x)
        expected[0, 0] = 1.5
        expected[0, 3] = 1.3
        expected[1, 2] = 1.2
        expected[1, 3] = 1.6
        result = kw(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertTrue(result.eq(expected).all())

        # Test values of updated duty cycle.
        new_duty = torch.tensor([1.0, 0, 1.0, 2.0, 0, 0]) / 2.0

        self.assertTrue(kw.duty_cycle.eq(new_duty).all())

        # Test forward with updated duty cycle.
        result = kw(x)

        expected = torch.zeros_like(x)
        expected[0, 1] = 1.0
        expected[0, 5] = 1.0
        expected[1, 1] = 1.0
        expected[1, 5] = 1.0

        self.assertEqual(result.shape, expected.shape)
        self.assertTrue(result.eq(expected).all())

    def test_k_winners_module_two(self):
        """
        Test a series of calls on the layer in training mode.
        """

        # Set up test input and module.
        x = self.x2
        n = 6

        expected = torch.zeros_like(x)
        expected[0, 0] = 1.5
        expected[0, 5] = 1.0
        expected[1, 2] = 1.2
        expected[1, 3] = 1.6

        kw = KWinners(
            n,
            percent_on=0.333,
            k_inference_factor=1.5,
            boost_strength=1.0,
            boost_strength_factor=0.5,
            duty_cycle_period=1000,
        )

        kw.train(mode=True)
        result = kw(x)
        result = kw(x)
        result = kw(x)
        result = kw(x)
        result = kw(x)
        result = kw(x)
        result = kw(x)

        self.assertTrue(result.eq(expected).all())

        # Test with mod.training = False.
        kw.train(mode=False)


if __name__ == "__main__":
    unittest.main()
