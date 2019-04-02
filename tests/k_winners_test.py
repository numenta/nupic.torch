# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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

from __future__ import print_function
import unittest

import torch
import nupic.torch.functions as F
from nupic.torch.modules import KWinners2d


class TestContext(object):
  def __init__(self):
    self.saved_tensors = None

  def save_for_backward(self,x):
    self.saved_tensors = (x,)


class KWinnersTest(unittest.TestCase):
  """

  """

  def setUp(self):
    # Tests will use 3 filters and image width, height = 2 X 2

    # Batch size 1
    x = torch.ones((1, 3, 2, 2))
    x[0, 0, 1, 0] = 1.1
    x[0, 0, 1, 1] = 1.2
    x[0, 1, 0, 1] = 1.2
    x[0, 2, 1, 0] = 1.3
    self.x = x
    self.gradient = torch.rand(x.shape)

    # Batch size 2
    x = torch.ones((2, 3, 2, 2))
    x[0, 0, 1, 0] = 1.1
    x[0, 0, 1, 1] = 1.2
    x[0, 1, 0, 1] = 1.2
    x[0, 2, 1, 0] = 1.3

    x[1, 0, 0, 0] = 1.4
    x[1, 1, 0, 0] = 1.5
    x[1, 1, 0, 1] = 1.6
    x[1, 2, 1, 1] = 1.7
    self.x2 = x
    self.gradient2 = torch.rand(x.shape)

    # All equal
    self.dutyCycle = torch.zeros((1, 3, 1, 1))
    self.dutyCycle[:] = 1.0 / 3.0


  def testOne(self):
    """
    Equal duty cycle, boost factor 0, k=4, batch size 1
    """
    x = self.x

    ctx = TestContext()

    result = F.k_winners2d.forward(ctx, x, self.dutyCycle, k=4, boostStrength=0.0)

    expected = torch.zeros_like(x)
    expected[0, 0, 1, 0] = 1.1
    expected[0, 0, 1, 1] = 1.2
    expected[0, 1, 0, 1] = 1.2
    expected[0, 2, 1, 0] = 1.3

    self.assertEqual(result.shape, expected.shape)

    numCorrect = (result == expected).sum()
    self.assertEqual(numCorrect, result.reshape(-1).size()[0])

    indices = ctx.saved_tensors[0].reshape(-1)
    expectedIndices = torch.tensor([2, 3, 10, 5])
    numCorrect = (indices == expectedIndices).sum()
    self.assertEqual(numCorrect, 4)

    # Test that gradient values are in the right places, that their sum is
    # equal, and that they have exactly the right number of nonzeros
    grad_x, _, _, _ = F.k_winners2d.backward(ctx, self.gradient)
    grad_x = grad_x.reshape(-1)
    self.assertEqual(
      (grad_x[indices] == self.gradient.reshape(-1)[indices]).sum(), 4)
    self.assertAlmostEqual(
      grad_x.sum(), self.gradient.reshape(-1)[indices].sum(), places=4)
    self.assertEqual(len(grad_x.nonzero()), 4)


  def testTwo(self):
    """
    Equal duty cycle, boost factor 0, k=3
    """
    x = self.x

    ctx = TestContext()

    result = F.k_winners2d.forward(ctx, x, self.dutyCycle, k=3, boostStrength=0.0)

    expected = torch.zeros_like(x)
    expected[0, 0, 1, 1] = 1.2
    expected[0, 1, 0, 1] = 1.2
    expected[0, 2, 1, 0] = 1.3

    self.assertEqual(result.shape, expected.shape)

    numCorrect = (result == expected).sum()
    self.assertEqual(numCorrect, result.reshape(-1).size()[0])

    indices = ctx.saved_tensors[0].reshape(-1)
    expectedIndices = torch.tensor([3, 10, 5])
    numCorrect = (indices == expectedIndices).sum()
    self.assertEqual(numCorrect, 3)

    # Test that gradient values are in the right places, that their sum is
    # equal, and that they have exactly the right number of nonzeros
    grad_x, _, _, _ = F.k_winners2d.backward(ctx, self.gradient)
    grad_x = grad_x.reshape(-1)
    self.assertEqual(
      (grad_x[indices] == self.gradient.reshape(-1)[indices]).sum(), 3)
    self.assertAlmostEqual(
      grad_x.sum().item(), self.gradient.reshape(-1)[indices].sum().item(), places=4)
    self.assertEqual(len(grad_x.nonzero()), 3)


  def testThree(self):
    """
    Equal duty cycle, boost factor=0, k=4, batch size=2
    """
    x = self.x2

    ctx = TestContext()

    result = F.k_winners2d.forward(ctx, x, self.dutyCycle, k=4, boostStrength=0.0)

    expected = torch.zeros_like(x)
    expected[0, 0, 1, 0] = 1.1
    expected[0, 0, 1, 1] = 1.2
    expected[0, 1, 0, 1] = 1.2
    expected[0, 2, 1, 0] = 1.3
    expected[1, 0, 0, 0] = 1.4
    expected[1, 1, 0, 0] = 1.5
    expected[1, 1, 0, 1] = 1.6
    expected[1, 2, 1, 1] = 1.7

    self.assertEqual(result.shape, expected.shape)

    numCorrect = (result == expected).sum()
    self.assertEqual(numCorrect, result.reshape(-1).size()[0])

    indices = ctx.saved_tensors[0]
    expectedIndices = torch.tensor([[2, 3, 10, 5], [0, 4, 5, 11]])
    numCorrect = (indices == expectedIndices).sum()
    self.assertEqual(numCorrect, 8)

    # Test that gradient values are in the right places, that their sum is
    # equal, and that they have exactly the right number of nonzeros
    out_grad, _, _, _ = F.k_winners2d.backward(ctx, self.gradient2)
    out_grad = out_grad.reshape(2, -1)
    in_grad = self.gradient2.reshape(2, -1)
    self.assertEqual((out_grad == in_grad).sum(), 8)
    self.assertEqual(len(out_grad.nonzero()), 8)


  def testFour(self):
    """
    Equal duty cycle, boost factor=0, k=3, batch size=2
    """
    x = self.x2

    ctx = TestContext()

    result = F.k_winners2d.forward(ctx, x, self.dutyCycle, k=3, boostStrength=0.0)

    expected = torch.zeros_like(x)
    expected[0, 0, 1, 1] = 1.2
    expected[0, 1, 0, 1] = 1.2
    expected[0, 2, 1, 0] = 1.3
    expected[1, 1, 0, 0] = 1.5
    expected[1, 1, 0, 1] = 1.6
    expected[1, 2, 1, 1] = 1.7

    self.assertEqual(result.shape, expected.shape)

    numCorrect = (result == expected).sum()
    self.assertEqual(numCorrect, result.reshape(-1).size()[0])

    indices = ctx.saved_tensors[0]
    expectedIndices = torch.tensor([[3, 10, 5], [4, 5, 11]])
    numCorrect = (indices == expectedIndices).sum()
    self.assertEqual(numCorrect, 6)

    # Test that gradient values are in the right places, that their sum is
    # equal, and that they have exactly the right number of nonzeros
    out_grad, _, _, _ = F.k_winners2d.backward(ctx, self.gradient2)
    out_grad = out_grad.reshape(2, -1)
    in_grad = self.gradient2.reshape(2, -1)
    self.assertEqual((out_grad == in_grad).sum(), 6)
    self.assertEqual(len(out_grad.nonzero()), 6)

  @unittest.skip("FIXME: Create test for KWinners2d module instead")
  def testDutyCycleUpdate(self):
    """
    Start with equal duty cycle, boost factor=0, k=4, batch size=2
    """
    x = self.x2

    expected = torch.zeros_like(x)
    expected[0, 0, 1, 0] = 1.1
    expected[0, 0, 1, 1] = 1.2
    expected[0, 1, 0, 1] = 1.2
    expected[0, 2, 1, 0] = 1.3
    expected[1, 0, 0, 0] = 1.4
    expected[1, 1, 0, 0] = 1.5
    expected[1, 1, 0, 1] = 1.6
    expected[1, 2, 1, 1] = 1.7

    dutyCycle = torch.zeros((1, 3, 1, 1))
    dutyCycle[:] = 1.0 / 3.0
    updateDutyCycleCNN(expected, dutyCycle, 2, 2)
    newDuty = torch.tensor([1.5000, 1.5000, 1.0000]) / 4.0
    diff = (dutyCycle.reshape(-1) - newDuty).abs().sum()
    self.assertLessEqual(diff, 0.001)

    dutyCycle[:] = 1.0 / 3.0
    updateDutyCycleCNN(expected, dutyCycle, 4, 4)
    newDuty = torch.tensor([0.3541667, 0.3541667, 0.2916667])
    diff = (dutyCycle.reshape(-1) - newDuty).abs().sum()
    self.assertLessEqual(diff, 0.001)


  def testKWinners2dModule(self):
    x = self.x2

    kw = KWinners2d(n=12, k=4, channels=3, kInferenceFactor=0.5,
                    boostStrength=1.0, boostStrengthFactor=0.5,
                    dutyCyclePeriod=1000)

    expected = torch.zeros_like(x)
    expected[0, 0, 1, 0] = 1.1
    expected[0, 0, 1, 1] = 1.2
    expected[0, 1, 0, 1] = 1.2
    expected[0, 2, 1, 0] = 1.3
    expected[1, 0, 0, 0] = 1.4
    expected[1, 1, 0, 0] = 1.5
    expected[1, 1, 0, 1] = 1.6
    expected[1, 2, 1, 1] = 1.7

    result = kw(x)
    self.assertEqual(result.shape, expected.shape)

    numCorrect = (result == expected).sum()
    self.assertEqual(numCorrect, result.reshape(-1).size()[0])

    newDuty = torch.tensor([1.5000, 1.5000, 1.0000]) / 4.0
    diff = (kw.dutyCycle.reshape(-1) - newDuty).abs().sum()
    self.assertLessEqual(diff, 0.001)


if __name__ == "__main__":
  unittest.main()
