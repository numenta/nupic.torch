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

import nupic.torch.functions as F
from nupic.torch.duty_cycle_metrics import binary_entropy, max_entropy


def update_boost_strength(m):
    """Function used to update KWinner modules boost strength. This is typically done
    during training at the beginning of each epoch.

    Call using :meth:`torch.nn.Module.apply` after each epoch if required
    For example: ``m.apply(update_boost_strength)``

    :param m: KWinner module
    """
    if isinstance(m, KWinnersBase):
        m.update_boost_strength()


class KWinnersBase(nn.Module, metaclass=abc.ABCMeta):
    """Base KWinners class.

    :param percent_on:
      The activity of the top k = percent_on * number of input units will be
      allowed to remain, the rest are set to zero.
    :type percent_on: float

    :param k_inference_factor:
      During inference (training=False) we increase percent_on by this factor.
      percent_on * k_inference_factor must be strictly less than 1.0, ideally much
      lower than 1.0
    :type k_inference_factor: float

    :param boost_strength:
      boost strength (0.0 implies no boosting). Must be >= 0.0
    :type boost_strength: float

    :param boost_strength_factor:
      Boost strength factor to use [0..1]
    :type boost_strength_factor: float

    :param duty_cycle_period:
      The period used to calculate duty cycles
    :type duty_cycle_period: int
    """

    def __init__(
        self,
        percent_on,
        k_inference_factor=1.0,
        boost_strength=1.0,
        boost_strength_factor=1.0,
        duty_cycle_period=1000,
    ):
        super(KWinnersBase, self).__init__()
        assert boost_strength >= 0.0
        assert 0.0 <= boost_strength_factor <= 1.0
        assert 0.0 < percent_on < 1.0
        assert 0.0 < percent_on * k_inference_factor < 1.0

        self.percent_on = percent_on
        self.percent_on_inference = percent_on * k_inference_factor
        self.k_inference_factor = k_inference_factor
        self.learning_iterations = 0
        self.n = 0
        self.k = 0
        self.k_inference = 0

        # Boosting related parameters. Put boost_strength in a buffer so that it
        # is saved in the state_dict. Keep a copy that remains a Python float so
        # that its value can be accessed in 'if' statements without blocking to
        # fetch from GPU memory.
        self.register_buffer("boost_strength", torch.tensor(boost_strength,
                                                            dtype=torch.float))
        self._cached_boost_strength = boost_strength

        self.boost_strength_factor = boost_strength_factor
        self.duty_cycle_period = duty_cycle_period

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._cached_boost_strength = self.boost_strength.item()

    def extra_repr(self):
        return (
            "n={0}, percent_on={1}, boost_strength={2}, boost_strength_factor={3}, "
            "k_inference_factor={4}, duty_cycle_period={5}".format(
                self.n, self.percent_on, self._cached_boost_strength,
                self.boost_strength_factor, self.k_inference_factor,
                self.duty_cycle_period
            )
        )

    @abc.abstractmethod
    def update_duty_cycle(self, x):
        r"""Updates our duty cycle estimates with the new value. Duty cycles are
        updated according to the following formula:

        .. math::
            dutyCycle = \frac{dutyCycle \times \left( period - batchSize \right)
                                + newValue}{period}

        :param x:
          Current activity of each unit
        """
        raise NotImplementedError

    def update_boost_strength(self):
        """Update boost strength by multiplying by the boost strength factor.
        This is typically done during training at the beginning of each epoch.
        """
        self._cached_boost_strength *= self.boost_strength_factor
        self.boost_strength.fill_(self._cached_boost_strength)

    def entropy(self):
        """Returns the current total entropy of this layer."""
        _, entropy = binary_entropy(self.duty_cycle)
        return entropy

    def max_entropy(self):
        """Returns the maximum total entropy we can expect from this layer."""
        return max_entropy(self.n, int(self.n * self.percent_on))


class KWinners(KWinnersBase):
    """Applies K-Winner function to the input tensor.

    See :class:`htmresearch.frameworks.pytorch.functions.k_winners`

    :param n:
      Number of units
    :type n: int

    :param percent_on:
      The activity of the top k = percent_on * n will be allowed to remain, the
      rest are set to zero.
    :type percent_on: float

    :param k_inference_factor:
      During inference (training=False) we increase percent_on by this factor.
      percent_on * k_inference_factor must be strictly less than 1.0, ideally much
      lower than 1.0
    :type k_inference_factor: float

    :param boost_strength:
      boost strength (0.0 implies no boosting).
    :type boost_strength: float

    :param boost_strength_factor:
      Boost strength factor to use [0..1]
    :type boost_strength_factor: float

    :param duty_cycle_period:
      The period used to calculate duty cycles
    :type duty_cycle_period: int

    :param break_ties:
        Whether to use a strict k-winners. Using break_ties=False is faster but
        may occasionally result in more than k active units.
    :type break_ties: bool

    :param relu:
        This will simulate the effect of having a ReLU before the KWinners.
    :type relu: bool

    :param inplace:
       Modify the input in-place.
    :type inplace: bool
    """

    def __init__(
        self,
        n,
        percent_on,
        k_inference_factor=1.5,
        boost_strength=1.0,
        boost_strength_factor=0.9,
        duty_cycle_period=1000,
        break_ties=False,
        relu=False,
        inplace=False,
    ):

        super(KWinners, self).__init__(
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
        )

        self.break_ties = break_ties
        self.inplace = inplace
        self.relu = relu

        self.n = n
        self.k = int(round(n * percent_on))
        self.k_inference = int(self.k * self.k_inference_factor)
        self.register_buffer("duty_cycle", torch.zeros(self.n))

    def forward(self, x):

        if self.training:
            x = F.kwinners(x, self.duty_cycle, self.k, self._cached_boost_strength,
                           self.break_ties, self.relu, self.inplace)
            self.update_duty_cycle(x)
        else:
            x = F.kwinners(x, self.duty_cycle, self.k_inference,
                           self._cached_boost_strength, self.break_ties, self.relu,
                           self.inplace)

        return x

    def update_duty_cycle(self, x):
        batch_size = x.shape[0]
        self.learning_iterations += batch_size
        period = min(self.duty_cycle_period, self.learning_iterations)
        self.duty_cycle.mul_(period - batch_size)
        self.duty_cycle.add_(x.gt(0).sum(dim=0, dtype=torch.float))
        self.duty_cycle.div_(period)

    def extra_repr(self):
        s = super().extra_repr()
        s += f", break_ties={self.break_ties}"
        if self.relu:
            s += ", relu=True"
        if self.inplace:
            s += ", inplace=True"
        return s


class KWinners2d(KWinnersBase):
    """
    Applies K-Winner function to the input tensor.

    See :class:`htmresearch.frameworks.pytorch.functions.k_winners2d`

    :param channels:
      Number of channels (filters) in the convolutional layer.
    :type channels: int

    :param percent_on:
      The activity of the top k = percent_on * number of input units will be
      allowed to remain, the rest are set to zero.
    :type percent_on: float

    :param k_inference_factor:
      During inference (training=False) we increase percent_on by this factor.
      percent_on * k_inference_factor must be strictly less than 1.0, ideally much
      lower than 1.0
    :type k_inference_factor: float

    :param boost_strength:
      boost strength (0.0 implies no boosting).
    :type boost_strength: float

    :param boost_strength_factor:
      Boost strength factor to use [0..1]
    :type boost_strength_factor: float

    :param duty_cycle_period:
      The period used to calculate duty cycles
    :type duty_cycle_period: int

    :param local:
        Whether or not to choose the k-winners locally (across the channels
        at each location) or globally (across the whole input and across
        all channels).
    :type local: bool

    :param break_ties:
        Whether to use a strict k-winners. Using break_ties=False is faster but
        may occasionally result in more than k active units.
    :type break_ties: bool

    :param relu:
        This will simulate the effect of having a ReLU before the KWinners.
    :type relu: bool

    :param inplace:
       Modify the input in-place.
    :type inplace: bool
    """

    def __init__(
        self,
        channels,
        percent_on=0.1,
        k_inference_factor=1.5,
        boost_strength=1.0,
        boost_strength_factor=0.9,
        duty_cycle_period=1000,
        local=False,
        break_ties=False,
        relu=False,
        inplace=False,
    ):

        super(KWinners2d, self).__init__(
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
        )

        self.channels = channels
        self.local = local
        self.break_ties = break_ties
        self.inplace = inplace
        self.relu = relu
        if local:
            self.k = int(round(self.channels * self.percent_on))
            self.k_inference = int(round(self.channels * self.percent_on_inference))

        self.register_buffer("duty_cycle", torch.zeros((1, channels, 1, 1)))

    def forward(self, x):

        if self.n == 0:
            self.n = np.prod(x.shape[1:])
            if not self.local:
                self.k = int(round(self.n * self.percent_on))
                self.k_inference = int(round(self.n * self.percent_on_inference))

        if self.training:
            x = F.kwinners2d(x, self.duty_cycle, self.k,
                             self._cached_boost_strength, self.local,
                             self.break_ties, self.relu, self.inplace)
            self.update_duty_cycle(x)
        else:
            x = F.kwinners2d(x, self.duty_cycle, self.k_inference,
                             self._cached_boost_strength, self.local,
                             self.break_ties, self.relu, self.inplace)

        return x

    def update_duty_cycle(self, x):
        batch_size = x.shape[0]
        self.learning_iterations += batch_size

        scale_factor = float(x.shape[2] * x.shape[3])
        period = min(self.duty_cycle_period, self.learning_iterations)
        self.duty_cycle.mul_(period - batch_size)
        s = x.gt(0).sum(dim=(0, 2, 3), dtype=torch.float) / scale_factor
        self.duty_cycle.reshape(-1).add_(s)
        self.duty_cycle.div_(period)

    def entropy(self):
        entropy = super(KWinners2d, self).entropy()
        return entropy * self.n / self.channels

    def extra_repr(self):
        s = (f"channels={self.channels}, local={self.local}"
             f", break_ties={self.break_ties}")
        if self.relu:
            s += ", relu=True"
        if self.inplace:
            s += ", inplace=True"
        s += ", {}".format(super().extra_repr())
        return s
