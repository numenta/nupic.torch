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


class KWinners(torch.autograd.Function):
    """A simple K-winner take all autograd function for creating layers with
    sparse output.

    .. note::
        Code adapted from this excellent tutorial:
        https://github.com/jcjohnson/pytorch-examples
    """

    @staticmethod
    def forward(ctx, x, duty_cycles, k, boost_strength):
        r"""Use the boost strength to compute a boost factor for each unit
        represented in x. These factors are used to increase the impact of each
        unit to improve their chances of being chosen. This encourages
        participation of more columns in the learning process.

        The boosting function is a curve defined as:

        .. math::
            boostFactors = \exp(-boostStrength \times (dutyCycles - targetDensity))

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

        :param ctx:
          Place where we can store information we will need to compute the gradients
          for the backward pass.

        :param x:
          Current activity of each unit, optionally batched along the 0th dimension.

        :param duty_cycles:
          The averaged duty cycle of each unit.

        :param k:
          The activity of the top k units will be allowed to remain, the rest are
          set to zero.

        :param boost_strength:
          A boost strength of 0.0 has no effect on x.

        :return:
          A tensor representing the activity of x after k-winner take all.
        """
        if boost_strength > 0.0:
            target_density = float(k) / x.size(1)
            boost_factors = torch.exp((target_density - duty_cycles) * boost_strength)
            boosted = x.detach() * boost_factors
        else:
            boosted = x.detach()

        # Take the boosted version of the input x, find the top k winners.
        # Compute an output that contains the values of x corresponding to the top k
        # boosted values
        topk, indices = boosted.topk(k, sorted=False)
        mask = torch.zeros_like(x).scatter(-1, indices, 1)
        res = mask * x
        ctx.save_for_backward(mask)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        """In the backward pass, we set the gradient to 1 for the winning
        units, and 0 for the others.
        """
        mask, = ctx.saved_tensors
        grad_x = grad_output * mask
        grad_x.requires_grad_(True)
        return grad_x, None, None, None


class KWinners2dGlobal(torch.autograd.Function):
    """
    A K-winner take all autograd function for CNN 2D inputs (batch, Channel, H, W)
    where the k-winners are chosen globally across the whole input

    .. seealso::      Function :class:`KWinners`
    """

    @staticmethod
    def forward(ctx, x, duty_cycles, k, boost_strength):
        """Use the boost strength to compute a boost factor for each unit
        represented in x. These factors are used to increase the impact of each
        unit to improve their chances of being chosen. This encourages
        participation of more columns in the learning process. See.

        :meth:`k_winners.forward` for more details.

        :param ctx:
          Place where we can store information we will need to compute the gradients
          for the backward pass.

        :param x:
          Current activity of each unit.

        :param duty_cycles:
          The averaged duty cycle of each unit.

        :param k:
          The activity of the top k units will be allowed to remain, the rest are
          set to zero.

        :param boost_strength:
          A boost strength of 0.0 has no effect on x.

        :return:
          A tensor representing the activity of x after k-winner take all.
        """
        batch_size = x.shape[0]
        if boost_strength > 0.0:
            target_density = float(k) / (x.shape[1] * x.shape[2] * x.shape[3])
            boost_factors = torch.exp((target_density - duty_cycles) * boost_strength)
            boosted = x.detach() * boost_factors
        else:
            boosted = x.detach()

        # Take the boosted version of the input x, find the top k winners.
        # Compute an output that only contains the values of x corresponding to
        # the top k boosted values. The rest of the elements in the output
        # should be 0.
        boosted = boosted.reshape((batch_size, -1))
        xr = x.reshape((batch_size, -1))
        res = torch.zeros_like(boosted)
        topk, indices = boosted.topk(k, dim=1, sorted=False)
        res.scatter_(1, indices, xr.gather(1, indices))
        res = res.reshape(x.shape)

        ctx.save_for_backward(indices)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        """In the backward pass, we set the gradient to 1 for the winning
        units, and 0 for the others.
        """
        batch_size = grad_output.shape[0]
        indices, = ctx.saved_tensors

        g = grad_output.reshape((batch_size, -1))
        grad_x = torch.zeros_like(g, requires_grad=False)
        grad_x.scatter_(1, indices, g.gather(1, indices))
        grad_x = grad_x.reshape(grad_output.shape)

        return grad_x, None, None, None


class KWinners2dLocal(torch.autograd.Function):
    """
    A K-winner take all autograd function for CNN 2D inputs (batch, Channel, H,
    W) where k-winners are chosen independently for each location. Local k-winners
    selects the top k channels locally for each of the H X W locations.

    .. seealso::      Function :class:`KWinners2dGlobal`
    """

    @staticmethod
    def forward(ctx, x, duty_cycles, k, boost_strength):
        """
          Take the boosted version of the input x, find the top k winners
          independently for each location. The output will only contain the
          values of x corresponding to the top k boosted values across all the
          channels at each location. The rest of the elements in the output
          should be 0.

        :param ctx:
          Place where we can store information we will need to compute the
          gradients for the backward pass.

        :param x:
          Current activity of each unit.

        :param duty_cycles:
          The averaged duty cycle of each unit.

        :param k:
          The activity of the top k units across the channels will be allowed to
          remain, the rest are set to zero.

        :param boost_strength:
          A boost strength of 0.0 has no effect on x.

        :return:
             A tensor representing the activity of x after k-winner take all.
        """
        batch_size, channels, h, w = x.shape
        if boost_strength > 0.0:
            # Apply boost strength to input computing density per channel
            target_density = float(k) / channels
            boost_factors = torch.exp((target_density - duty_cycles) * boost_strength)
            boosted = x.detach() * boost_factors
        else:
            boosted = x.detach()

        # Select top K channels from the boosted values
        topk, indices = boosted.topk(k=k, dim=1)
        res = torch.zeros_like(x)
        res.scatter_(1, indices, x.gather(1, indices))

        ctx.save_for_backward(indices)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, we set the gradient to 1 for the winning units,
        and 0 for the others.
        """
        indices, = ctx.saved_tensors
        grad_x = torch.zeros_like(grad_output, requires_grad=False)
        grad_x.scatter_(1, indices, grad_output.gather(1, indices))
        return grad_x, None, None, None, None
