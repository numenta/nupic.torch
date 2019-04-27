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
import abc

import numpy as np
import torch
import torch.nn as nn

from nupic.torch.duty_cycle_metrics import maxEntropy, binaryEntropy
from nupic.torch.functions import k_winners, k_winners2d



def updateBoostStrength(m):
  """
  Function used to update KWinner modules boost strength after each epoch.

  Call using :meth:`torch.nn.Module.apply` after each epoch if required
  For example: ``m.apply(updateBoostStrength)``

  :param m: KWinner module
  """
  if isinstance(m, KWinnersBase):
    if m.training:
      m.boostStrength = m.boostStrength * m.boostStrengthFactor



class KWinnersBase(nn.Module):
  """
  Base KWinners class
  """
  __metaclass__ = abc.ABCMeta


  def __init__(self, percent_on, kInferenceFactor=1.0, boostStrength=1.0,
               boostStrengthFactor=1.0, dutyCyclePeriod=1000):
    """
    :param percent_on:
      The activity of the top k = percent_on * number of input units will be
      allowed to remain, the rest are set to zero.
    :type percent_on: float

    :param kInferenceFactor:
      During inference (training=False) we increase percent_on by this factor.
      percent_on * kInferenceFactor must be strictly less than 1.0, ideally much
      lower than 1.0
    :type kInferenceFactor: float

    :param boostStrength:
      boost strength (0.0 implies no boosting). Must be >= 0.0
    :type boostStrength: float

    :param boostStrengthFactor:
      Boost strength factor to use [0..1]
    :type boostStrengthFactor: float

    :param dutyCyclePeriod:
      The period used to calculate duty cycles
    :type dutyCyclePeriod: int
    """
    super(KWinnersBase, self).__init__()
    assert (boostStrength >= 0.0)
    assert (0.0 <= boostStrengthFactor <= 1.0)
    assert (0.0 < percent_on < 1.0)
    assert (0.0 < percent_on * kInferenceFactor < 1.0)


    self.percent_on = percent_on
    self.percent_on_inference = percent_on * kInferenceFactor
    self.kInferenceFactor = kInferenceFactor
    self.learningIterations = 0
    self.n = 0
    self.k = 0
    self.k_inference = 0

    # Boosting related parameters
    self.boostStrength = boostStrength
    self.boostStrengthFactor = boostStrengthFactor
    self.dutyCyclePeriod = dutyCyclePeriod


  def extra_repr(self):
    return 'n={}, percent_on={}, boostStrength={}, dutyCyclePeriod={}'.format(
      self.n, self.percent_on, self.boostStrength, self.dutyCyclePeriod)


  def getLearningIterations(self):
    return self.learningIterations


  @abc.abstractmethod
  def updateDutyCycle(self, x):
    """
     Updates our duty cycle estimates with the new value. Duty cycles are
     updated according to the following formula:

    .. math::
        dutyCycle = \\frac{dutyCycle \\times \\left( period - batchSize \\right)
                            + newValue}{period}
    :param x:
      Current activity of each unit
    """
    raise NotImplementedError


  def updateBoostStrength(self):
    """
    Update boost strength using given strength factor during training
    """
    if self.training:
      self.boostStrength = self.boostStrength * self.boostStrengthFactor


  def entropy(self):
    """
    Returns the current total entropy of this layer
    """
    _, entropy = binaryEntropy(self.dutyCycle)
    return entropy


  def maxEntropy(self):
    """
    Returns the maximum total entropy we can expect from this layer
    """
    return maxEntropy(self.n, int(self.n * self.percent_on))



class KWinners(KWinnersBase):
  """
  Applies K-Winner function to the input tensor

  See :class:`htmresearch.frameworks.pytorch.functions.k_winners`

  """


  def __init__(self, n, percent_on, kInferenceFactor=1.0, boostStrength=1.0,
               boostStrengthFactor=1.0, dutyCyclePeriod=1000):
    """
    :param n:
      Number of units
    :type n: int

    :param percent_on:
      The activity of the top k = percent_on * n will be allowed to remain, the
      rest are set to zero.
    :type percent_on: float

    :param kInferenceFactor:
      During inference (training=False) we increase percent_on by this factor.
      percent_on * kInferenceFactor must be strictly less than 1.0, ideally much
      lower than 1.0
    :type kInferenceFactor: float

    :param boostStrength:
      boost strength (0.0 implies no boosting).
    :type boostStrength: float

    :param boostStrengthFactor:
      Boost strength factor to use [0..1]
    :type boostStrengthFactor: float

    :param dutyCyclePeriod:
      The period used to calculate duty cycles
    :type dutyCyclePeriod: int
    """

    super(KWinners, self).__init__(percent_on=percent_on,
                                   kInferenceFactor=kInferenceFactor,
                                   boostStrength=boostStrength,
                                   boostStrengthFactor=boostStrengthFactor,
                                   dutyCyclePeriod=dutyCyclePeriod)
    self.n = n
    self.k = int(round(n * percent_on))
    self.k_inference = int(self.k * self.kInferenceFactor)
    self.register_buffer("dutyCycle", torch.zeros(self.n))


  def forward(self, x):

    if self.training:
      x = k_winners.apply(x, self.dutyCycle, self.k, self.boostStrength)
      self.updateDutyCycle(x)
    else:
      x = k_winners.apply(x, self.dutyCycle, self.k_inference, self.boostStrength)

    return x


  def updateDutyCycle(self, x):
    batchSize = x.shape[0]
    self.learningIterations += batchSize
    period = min(self.dutyCyclePeriod, self.learningIterations)
    self.dutyCycle.mul_(period - batchSize)
    self.dutyCycle.add_(x.gt(0).sum(dim=0, dtype=torch.float))
    self.dutyCycle.div_(period)




class KWinners2d(KWinnersBase):
  """
  Applies K-Winner function to the input tensor

  See :class:`htmresearch.frameworks.pytorch.functions.k_winners2d`

  """


  def __init__(self, channels, percent_on=0.1, kInferenceFactor=1.0,
               boostStrength=1.0, boostStrengthFactor=1.0,
               dutyCyclePeriod=1000):
    """

    :param channels:
      Number of channels (filters) in the convolutional layer.
    :type channels: int

    :param percent_on:
      The activity of the top k = percent_on * number of input units will be
      allowed to remain, the rest are set to zero.
    :type percent_on: float

    :param kInferenceFactor:
      During inference (training=False) we increase percent_on by this factor.
      percent_on * kInferenceFactor must be strictly less than 1.0, ideally much
      lower than 1.0
    :type kInferenceFactor: float

    :param boostStrength:
      boost strength (0.0 implies no boosting).
    :type boostStrength: float

    :param boostStrengthFactor:
      Boost strength factor to use [0..1]
    :type boostStrengthFactor: float

    :param dutyCyclePeriod:
      The period used to calculate duty cycles
    :type dutyCyclePeriod: int
    """
    super(KWinners2d, self).__init__(percent_on=percent_on,
                                     kInferenceFactor=kInferenceFactor,
                                     boostStrength=boostStrength,
                                     boostStrengthFactor=boostStrengthFactor,
                                     dutyCyclePeriod=dutyCyclePeriod)

    self.channels = channels
    self.register_buffer("dutyCycle", torch.zeros((1, channels, 1, 1)))


  def forward(self, x):

    if self.n == 0:
      self.n = np.prod(x.shape[1:])
      self.k = int(round(self.n * self.percent_on))
      self.k_inference = int(round(self.n * self.percent_on_inference))

    if self.training:
      x = k_winners2d.apply(x, self.dutyCycle, self.k, self.boostStrength)
      self.updateDutyCycle(x)

    else:
      x = k_winners2d.apply(x, self.dutyCycle, self.k_inference, self.boostStrength)

    return x


  def updateDutyCycle(self, x):
    batchSize = x.shape[0]
    self.learningIterations += batchSize

    scaleFactor = float(x.shape[2] * x.shape[3])
    period = min(self.dutyCyclePeriod, self.learningIterations)
    self.dutyCycle.mul_(period - batchSize)
    s = x.gt(0).sum(dim=(0, 2, 3), dtype=torch.float) / scaleFactor
    self.dutyCycle.reshape(-1).add_(s)
    self.dutyCycle.div_(period)


  def entropy(self):
    entropy = super(KWinners2d, self).entropy()
    return entropy * self.n / self.channels

