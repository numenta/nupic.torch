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
import torch


def kwinners(x, duty_cycles, k, boost_strength, break_ties=False, relu=False,
             inplace=False):
    """
    A simple K-winner take all function for creating layers with sparse output.

    Use the boost strength to compute a boost factor for each unit represented
    in x. These factors are used to increase the impact of each unit to improve
    their chances of being chosen. This encourages participation of more columns
    in the learning process.

    The boosting function is a curve defined as:

    .. math::
        boostFactors = \\exp(-boostStrength \\times (dutyCycles - targetDensity))

    Intuitively this means that units that have been active (i.e. in the top-k)
    at the target activation level have a boost factor of 1, meaning their
    activity is not boosted. Columns whose duty cycle drops too much below that
    of their neighbors are boosted depending on how infrequently they have been
    active. Unit that has been active more than the target activation level
    have a boost factor below 1, meaning their activity is suppressed and
    they are less likely to be in the top-k.

    Note that we do not transmit the boosted values. We only use boosting to
    determine the winning units.

    The target activation density for each unit is k / number of units. The
    boostFactor depends on the duty_cycles via an exponential function::

            boostFactor
                ^
                |
                |\
                | \
          1  _  |  \
                |    _
                |      _ _
                |          _ _ _ _
                +--------------------> duty_cycles
                   |
              target_density

    :param x:
      Current activity of each unit, optionally batched along the 0th dimension.

    :param duty_cycles:
      The averaged duty cycle of each unit.

    :param k:
      The activity of the top k units will be allowed to remain, the rest are
      set to zero.

    :param boost_strength:
      A boost strength of 0.0 has no effect on x.

    :param break_ties:
      Whether to use a strict k-winners. Using break_ties=False is faster but
      may occasionally result in more than k active units.

    :param relu:
      Whether to simulate the effect of applying ReLU before KWinners

    :param inplace:
      Whether to modify x in place

    :return:
      A tensor representing the activity of x after k-winner take all.
    """
    if k == 0:
        return torch.zeros_like(x)

    boosted = boost_activations(x, duty_cycles, boost_strength)

    if break_ties:
        indices = boosted.topk(k=k, dim=1, sorted=False)[1]
        off_mask = torch.ones_like(boosted, dtype=torch.bool)
        off_mask.scatter_(1, indices, False)

        if relu:
            off_mask |= (boosted <= 0)
    else:
        threshold = boosted.kthvalue(x.shape[1] - k + 1, dim=1,
                                     keepdim=True)[0]

        if relu:
            threshold.clamp_(min=0)
        off_mask = boosted < threshold

    if inplace:
        return x.masked_fill_(off_mask, 0)
    else:
        return x.masked_fill(off_mask, 0)


def kwinners2d(x, duty_cycles, k, boost_strength, local=True, break_ties=False,
               relu=False, inplace=False):
    """
    A K-winner take all function for creating Conv2d layers with sparse output.

    If local=True, k-winners are chosen independently for each location. For
    Conv2d inputs (batch, channel, H, W), the top k channels are selected
    locally for each of the H X W locations. If there is a tie for the kth
    highest boosted value, there will be more than k winners.

    The boost strength is used to compute a boost factor for each unit
    represented in x. These factors are used to increase the impact of each unit
    to improve their chances of being chosen. This encourages participation of
    more columns in the learning process. See :meth:`kwinners` for more details.

    :param x:
      Current activity of each unit.

    :param duty_cycles:
      The averaged duty cycle of each unit.

    :param k:
      The activity of the top k units across the channels will be allowed to
      remain, the rest are set to zero.

    :param boost_strength:
      A boost strength of 0.0 has no effect on x.

    :param local:
      Whether or not to choose the k-winners locally (across the channels at
      each location) or globally (across the whole input and across all
      channels).

    :param break_ties:
      Whether to use a strict k-winners. Using break_ties=False is faster but
      may occasionally result in more than k active units.

    :param relu:
      Whether to simulate the effect of applying ReLU before KWinners.

    :param inplace:
      Whether to modify x in place

    :return:
         A tensor representing the activity of x after k-winner take all.
    """
    if k == 0:
        return torch.zeros_like(x)

    boosted = boost_activations(x, duty_cycles, boost_strength)

    if break_ties:
        if local:
            indices = boosted.topk(k=k, dim=1, sorted=False)[1]
            off_mask = torch.ones_like(boosted, dtype=torch.bool)
            off_mask.scatter_(1, indices, False)
        else:
            shape2 = (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
            indices = boosted.view(shape2).topk(k, dim=1, sorted=False)[1]
            off_mask = torch.ones(shape2, dtype=torch.bool, device=x.device)
            off_mask.scatter_(1, indices, False)
            off_mask = off_mask.view(x.shape)

        if relu:
            off_mask |= (boosted <= 0)
    else:
        if local:
            threshold = boosted.kthvalue(x.shape[1] - k + 1, dim=1,
                                         keepdim=True)[0]
        else:
            threshold = boosted.view(x.shape[0], -1).kthvalue(
                x.shape[1] * x.shape[2] * x.shape[3] - k + 1, dim=1)[0]
            threshold = threshold.view(x.shape[0], 1, 1, 1)

        if relu:
            threshold.clamp_(min=0)
        off_mask = boosted < threshold

    if inplace:
        return x.masked_fill_(off_mask, 0)
    else:
        return x.masked_fill(off_mask, 0)


def boost_activations(x, duty_cycles, boost_strength):
    """
    Boosting as documented in :meth:`kwinners` would compute
      x * torch.exp((target_density - duty_cycles) * boost_strength)
    but instead we compute
      x * torch.exp(-boost_strength * duty_cycles)
    which is equal to the former value times a positive constant, so it will
    have the same ranked order.

    :param x:
      Current activity of each unit.

    :param duty_cycles:
      The averaged duty cycle of each unit.

    :param boost_strength:
      A boost strength of 0.0 has no effect on x.

    :return:
         A tensor representing the boosted activity
    """
    if boost_strength > 0.0:
        return x.detach() * torch.exp(-boost_strength * duty_cycles)
    else:
        return x.detach()


__all__ = [
    "kwinners",
    "kwinners2d",
]
