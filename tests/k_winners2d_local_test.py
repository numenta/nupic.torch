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
from torch.autograd.gradcheck import gradcheck

from nupic.torch.modules import KWinners2d


class KWinner2dLocalTest(unittest.TestCase):
    def setUp(self):
        x = torch.zeros(3, 4, 2, 2)
        x[0, :, 0, 0] = torch.FloatTensor([1, 2, 3, 4])
        x[0, :, 0, 1] = torch.FloatTensor([2, 3, 0, 6])
        x[0, :, 1, 0] = torch.FloatTensor([-1, -2, -3, -4])
        x[0, :, 1, 1] = torch.FloatTensor([10, 11, 12, 13])

        x[1, :, 0, 0] = torch.FloatTensor([10, 12, 31, 42])
        x[1, :, 0, 1] = torch.FloatTensor([0, 1, 0, 6])
        x[1, :, 1, 0] = torch.FloatTensor([-2, -10, -11, -4])
        x[1, :, 1, 1] = torch.FloatTensor([7, 1, 10, 3])

        self.x = x

    def test_k_winners2d_one(self):
        """
        Equal duty cycle, boost_strength=0, percent_on=0.5, batch size=1
        """
        x = self.x[0:1]
        n, c, h, w = x.shape
        kw = KWinners2d(
            percent_on=0.5,  # k=2
            channels=c,
            k_inference_factor=1.0,
            boost_strength=0.0,
            duty_cycle_period=1000,
            local=True
        )
        kw.train(mode=False)

        expected = torch.zeros_like(x)

        expected[0, [2, 3], 0, 0] = x[0, [2, 3], 0, 0]
        expected[0, [1, 3], 0, 1] = x[0, [1, 3], 0, 1]
        expected[0, [0, 1], 1, 0] = x[0, [0, 1], 1, 0]
        expected[0, [2, 3], 1, 1] = x[0, [2, 3], 1, 1]

        result = kw(x)
        self.assertEqual(result.shape, expected.shape)

        num_correct = (result == expected).sum()
        self.assertEqual(num_correct, result.reshape(-1).size()[0])

    def test_k_winners2d_two(self):
        """
        Equal duty cycle, boost_strength=0, percent_on=0.5, batch size=2
        """
        x = self.x[0:2]
        n, c, h, w = x.shape
        kw = KWinners2d(
            percent_on=0.5,  # k=2
            channels=c,
            k_inference_factor=1.0,
            boost_strength=0.0,
            duty_cycle_period=1000,
            local=True
        )
        kw.train(mode=False)
        expected = torch.zeros_like(x)

        expected[0, [2, 3], 0, 0] = x[0, [2, 3], 0, 0]
        expected[0, [1, 3], 0, 1] = x[0, [1, 3], 0, 1]
        expected[0, [0, 1], 1, 0] = x[0, [0, 1], 1, 0]
        expected[0, [2, 3], 1, 1] = x[0, [2, 3], 1, 1]

        expected[1, [2, 3], 0, 0] = x[1, [2, 3], 0, 0]
        expected[1, [1, 3], 0, 1] = x[1, [1, 3], 0, 1]
        expected[1, [0, 3], 1, 0] = x[1, [0, 3], 1, 0]
        expected[1, [0, 2], 1, 1] = x[1, [0, 2], 1, 1]

        result = kw(x)
        self.assertEqual(result.shape, expected.shape)

        num_correct = (result == expected).sum()
        self.assertEqual(num_correct, result.reshape(-1).size()[0])

    def test_k_winners2d_train(self):
        """
        Test training
        Changing duty cycle, boost_strength=1, percent_on=0.5, batch size=2
        """
        x = self.x[0:2]
        n, c, h, w = x.shape
        kw = KWinners2d(
            percent_on=0.5,
            channels=c,
            boost_strength=1.0,
            duty_cycle_period=10,
            local=True
        )

        kw.train(mode=True)

        # Expectation due to boosting after the second training step
        expected = torch.zeros_like(x)
        expected[0, [2, 3], 0, 0] = x[0, [2, 3], 0, 0]
        expected[0, [1, 3], 0, 1] = x[0, [1, 3], 0, 1]
        expected[0, [0, 1], 1, 0] = x[0, [0, 1], 1, 0]
        expected[0, [0, 1], 1, 1] = x[0, [0, 1], 1, 1]

        expected[1, [2, 3], 0, 0] = x[1, [2, 3], 0, 0]
        expected[1, [1, 3], 0, 1] = x[1, [1, 3], 0, 1]
        expected[1, [0, 3], 1, 0] = x[1, [0, 3], 1, 0]
        expected[1, [0, 2], 1, 1] = x[1, [0, 2], 1, 1]

        result = kw(x)
        result = kw(x)
        self.assertTrue(result.eq(expected).all())

        # Expectation due to boosting after the fourth training step
        expected_boosted = expected.clone()
        expected_boosted[0, [0, 1], 1, 1] = 0
        expected_boosted[0, [0, 2], 1, 1] = x[0, [0, 2], 1, 1]

        result = kw(x)
        result = kw(x)
        self.assertTrue(result.eq(expected_boosted).all())

    def test_k_winners2d_grad(self):
        """
        Test gradient
        """
        x = torch.randn(self.x.size(), dtype=torch.double, requires_grad=True)
        n, c, h, w = x.shape
        kw = KWinners2d(
            percent_on=0.5,
            channels=c,
            k_inference_factor=1.0,
            boost_strength=0.0,
            boost_strength_factor=1.0,
            duty_cycle_period=1000,
            local=True
        )
        self.assertTrue(gradcheck(kw, x, raise_exception=True))


if __name__ == "__main__":
    unittest.main()
