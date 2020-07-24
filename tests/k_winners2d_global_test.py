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

        expected = torch.zeros_like(x)
        expected[0, 0, 1, 0] = x[0, 0, 1, 0]
        expected[0, 0, 1, 1] = x[0, 0, 1, 1]
        expected[0, 1, 0, 1] = x[0, 1, 0, 1]
        expected[0, 2, 1, 0] = x[0, 2, 1, 0]

        for break_ties in [True, False]:
            with self.subTest(break_ties=break_ties):
                result = F.kwinners2d(x, self.duty_cycle, k=4,
                                      boost_strength=0.0, local=False,
                                      break_ties=break_ties)

                self.assertEqual(result.shape, expected.shape)

                num_correct = (result == expected).sum()
                self.assertEqual(num_correct, result.reshape(-1).size()[0])

    def test_two(self):
        """Equal duty cycle, boost factor 0, k=3."""
        x = self.x

        expected = torch.zeros_like(x)
        expected[0, 0, 1, 1] = x[0, 0, 1, 1]
        expected[0, 1, 0, 1] = x[0, 1, 0, 1]
        expected[0, 2, 1, 0] = x[0, 2, 1, 0]

        for break_ties in [True, False]:
            with self.subTest(break_ties=break_ties):
                result = F.kwinners2d(x, self.duty_cycle, k=3,
                                      boost_strength=0.0, local=False,
                                      break_ties=break_ties)

                self.assertEqual(result.shape, expected.shape)

                num_correct = (result == expected).sum()
                self.assertEqual(num_correct, result.reshape(-1).size()[0])

    def test_three(self):
        """Equal duty cycle, boost factor=0, k=4, batch size=2."""
        x = self.x2

        expected = torch.zeros_like(x)
        expected[0, 0, 1, 0] = x[0, 0, 1, 0]
        expected[0, 0, 1, 1] = x[0, 0, 1, 1]
        expected[0, 1, 0, 1] = x[0, 1, 0, 1]
        expected[0, 2, 1, 0] = x[0, 2, 1, 0]
        expected[1, 0, 0, 0] = x[1, 0, 0, 0]
        expected[1, 1, 0, 0] = x[1, 1, 0, 0]
        expected[1, 1, 0, 1] = x[1, 1, 0, 1]
        expected[1, 2, 1, 1] = x[1, 2, 1, 1]

        for break_ties in [True, False]:
            with self.subTest(break_ties=break_ties):
                result = F.kwinners2d(x, self.duty_cycle, k=4,
                                      boost_strength=0.0, local=False,
                                      break_ties=break_ties)
                self.assertEqual(result.shape, expected.shape)

                num_correct = (result == expected).sum()
                self.assertEqual(num_correct, result.reshape(-1).size()[0])

    def test_four(self):
        """Equal duty cycle, boost factor=0, k=3, batch size=2."""
        x = self.x2

        expected = torch.zeros_like(x)
        expected[0, 0, 1, 1] = x[0, 0, 1, 1]
        expected[0, 1, 0, 1] = x[0, 1, 0, 1]
        expected[0, 2, 1, 0] = x[0, 2, 1, 0]
        expected[1, 1, 0, 0] = x[1, 1, 0, 0]
        expected[1, 1, 0, 1] = x[1, 1, 0, 1]
        expected[1, 2, 1, 1] = x[1, 2, 1, 1]

        for break_ties in [True, False]:
            with self.subTest(break_ties=break_ties):
                result = F.kwinners2d(x, self.duty_cycle, k=3,
                                      boost_strength=0.0, local=False,
                                      break_ties=break_ties)

                self.assertEqual(result.shape, expected.shape)

                num_correct = (result == expected).sum()
                self.assertEqual(num_correct, result.reshape(-1).size()[0])

    def test_k_winners2d_module_one(self):
        x = self.x2

        expected = torch.zeros_like(x)
        expected[0, 0, 1, 0] = x[0, 0, 1, 0]
        expected[0, 0, 1, 1] = x[0, 0, 1, 1]
        expected[0, 1, 0, 1] = x[0, 1, 0, 1]
        expected[0, 2, 1, 0] = x[0, 2, 1, 0]
        expected[1, 0, 0, 0] = x[1, 0, 0, 0]
        expected[1, 1, 0, 0] = x[1, 1, 0, 0]
        expected[1, 1, 0, 1] = x[1, 1, 0, 1]
        expected[1, 2, 1, 1] = x[1, 2, 1, 1]

        for break_ties in [True, False]:
            with self.subTest(break_ties=break_ties):
                kw = KWinners2d(
                    percent_on=0.333,
                    channels=3,
                    k_inference_factor=0.5,
                    boost_strength=1.0,
                    boost_strength_factor=0.5,
                    duty_cycle_period=1000,
                    local=False,
                    break_ties=break_ties,
                )

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

        for break_ties in [True, False]:
            with self.subTest(break_ties=break_ties):
                kw = KWinners2d(
                    percent_on=0.25,
                    channels=3,
                    k_inference_factor=0.5,
                    boost_strength=1.0,
                    boost_strength_factor=0.5,
                    duty_cycle_period=1000,
                    local=False,
                    break_ties=break_ties,
                )

                kw.train(mode=True)
                result = kw(x)
                result = kw(x)
                result = kw(x)
                result = kw(x)
                result = kw(x)
                result = kw(x)

                self.assertTrue(result.eq(expected).all())

    def test_k_winners2d_relu(self):
        x = torch.zeros(2, 4, 2, 2)
        x[0, :, 0, 0] = torch.FloatTensor([-6, -7, -8, -9])
        x[0, :, 0, 1] = torch.FloatTensor([0, 3, -42, -19])
        x[0, :, 1, 0] = torch.FloatTensor([-1, -2, 3, -4])
        x[0, :, 1, 1] = torch.FloatTensor([-10, -11, -12, -13])

        x[1, :, 0, 0] = torch.FloatTensor([-10, -12, -31, -42])
        x[1, :, 0, 1] = torch.FloatTensor([0, -1, 0, -6])
        x[1, :, 1, 0] = torch.FloatTensor([-2, -10, -11, -4])
        x[1, :, 1, 1] = torch.FloatTensor([-7, -1, -10, -3])

        expected = torch.zeros(2, 4, 2, 2)
        expected[0, 1, 0, 1] = 3
        expected[0, 2, 1, 0] = 3

        for break_ties in [True, False]:
            with self.subTest(break_ties=break_ties):
                kw = KWinners2d(
                    percent_on=0.25,
                    channels=4,
                    k_inference_factor=0.5,
                    boost_strength=1.0,
                    boost_strength_factor=0.5,
                    duty_cycle_period=1000,
                    local=False,
                    break_ties=break_ties,
                    relu=True,
                )

                result = kw(x)
                self.assertTrue(result.eq(expected).all())

    def test_k_winners2d_global_grad(self):
        """
        Test gradient
        """
        x = self.x2.clone().requires_grad_(True)
        n, c, h, w = x.shape

        grad = self.gradient2

        expected = torch.zeros_like(x)
        expected[0, 0, 1, 0] = grad[0, 0, 1, 0]
        expected[0, 0, 1, 1] = grad[0, 0, 1, 1]
        expected[0, 1, 0, 1] = grad[0, 1, 0, 1]
        expected[0, 2, 1, 0] = grad[0, 2, 1, 0]

        expected[1, 0, 0, 0] = grad[1, 0, 0, 0]
        expected[1, 1, 0, 0] = grad[1, 1, 0, 0]
        expected[1, 1, 0, 1] = grad[1, 1, 0, 1]
        expected[1, 2, 1, 1] = grad[1, 2, 1, 1]

        for break_ties in [True, False]:
            with self.subTest(break_ties=break_ties):
                kw = KWinners2d(
                    percent_on=(1 / 3),
                    channels=c,
                    k_inference_factor=1.0,
                    boost_strength=0.0,
                    duty_cycle_period=1000,
                    local=False,
                    break_ties=break_ties,
                )
                kw.train(mode=True)
                y = kw(x)
                y.backward(grad)
                torch.testing.assert_allclose(x.grad, expected)
                x.grad.zero_()


if __name__ == "__main__":
    unittest.main()
