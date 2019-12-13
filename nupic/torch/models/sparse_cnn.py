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
from collections import OrderedDict

from torch import nn
from torch.hub import load_state_dict_from_url

from nupic.torch.modules import (
    Flatten,
    KWinners,
    KWinners2d,
    SparseWeights,
    SparseWeights2d,
)


class MNISTSparseCNN(nn.Sequential):
    """Sparse CNN model used to classify `MNIST` dataset as described in `How
    Can We Be So Dense?`_ paper.

    .. _`How Can We Be So Dense?`: https://arxiv.org/abs/1903.11257

    :param cnn_out_channels: output channels for each CNN layer
    :param cnn_percent_on: Percent of units allowed to remain on each convolution
                           layer
    :param cnn_weight_sparsity: Percent of weights that are allowed to be non-zero
    :param linear_units: Number of units in the linear layer
    :param linear_percent_on: Percent of units allowed to remain on the linear
                              layer
    :param linear_weight_sparsity: Percent of weights that are allowed to be
                                   non-zero in the linear layer
    :param k_inference_factor: During inference (training=False) we increase
                               `percent_on` in all sparse layers by this factor
    :param boost_strength: boost strength (0.0 implies no boosting)
    :param boost_strength_factor: Boost strength factor to use [0..1]
    :param duty_cycle_period: The period used to calculate duty cycles
    :param kwinner_local: Whether or not to choose the k-winners locally (across
                          the channels at each location) or globally (across the
                          whole input and across all channels).
    """

    def __init__(self,
                 cnn_out_channels=(32, 64),
                 cnn_percent_on=(0.1, 0.2),
                 cnn_weight_sparsity=(0.6, 0.45),
                 linear_units=700,
                 linear_percent_on=0.2,
                 linear_weight_sparsity=0.2,
                 boost_strength=1.5,
                 boost_strength_factor=0.85,
                 k_inference_factor=1.0,
                 duty_cycle_period=1000,
                 kwinner_local=False
                 ):
        super(MNISTSparseCNN, self).__init__(OrderedDict([
            # First Sparse CNN layer
            ("cnn1", SparseWeights2d(nn.Conv2d(1, cnn_out_channels[0], 5),
                                     cnn_weight_sparsity[0])),
            ("cnn1_maxpool", nn.MaxPool2d(2)),
            ("cnn1_kwinner", KWinners2d(channels=cnn_out_channels[0],
                                        percent_on=cnn_percent_on[0],
                                        k_inference_factor=k_inference_factor,
                                        boost_strength=boost_strength,
                                        boost_strength_factor=boost_strength_factor,
                                        duty_cycle_period=duty_cycle_period,
                                        local=kwinner_local)),

            # Second Sparse CNN layer
            ("cnn2", SparseWeights2d(nn.Conv2d(cnn_out_channels[0],
                                               cnn_out_channels[1], 5),
                                     cnn_weight_sparsity[1])),
            ("cnn2_maxpool", nn.MaxPool2d(2)),
            ("cnn2_kwinner", KWinners2d(channels=cnn_out_channels[1],
                                        percent_on=cnn_percent_on[1],
                                        k_inference_factor=k_inference_factor,
                                        boost_strength=boost_strength,
                                        boost_strength_factor=boost_strength_factor,
                                        duty_cycle_period=duty_cycle_period,
                                        local=kwinner_local)),

            ("flatten", Flatten()),

            # Sparse Linear layer
            ("linear", SparseWeights(
                nn.Linear(16 * cnn_out_channels[1], linear_units),
                weight_sparsity=linear_weight_sparsity)),
            ("linear_kwinner", KWinners(n=linear_units,
                                        percent_on=linear_percent_on,
                                        k_inference_factor=k_inference_factor,
                                        boost_strength=boost_strength,
                                        boost_strength_factor=boost_strength_factor,
                                        duty_cycle_period=duty_cycle_period)),

            # Classifier
            ("output", nn.Linear(linear_units, 10)),
            ("softmax", nn.LogSoftmax(dim=1))
        ]))


class GSCSparseCNN(nn.Sequential):
    """Sparse CNN model used to classify `Google Speech Commands` dataset as
    described in `How Can We Be So Dense?`_ paper.

    .. _`How Can We Be So Dense?`: https://arxiv.org/abs/1903.11257

    :param cnn_out_channels: output channels for each CNN layer
    :param cnn_percent_on: Percent of units allowed to remain on each convolution
                           layer
    :param cnn_weight_sparsity: Percent of weights that are allowed to be non-zero
    :param linear_units: Number of units in the linear layer
    :param linear_percent_on: Percent of units allowed to remain on the linear
                              layer
    :param linear_weight_sparsity: Percent of weights that are allowed to be
                                   non-zero in the linear layer
    :param k_inference_factor: During inference (training=False) we increase
                               `percent_on` in all sparse layers by this factor
    :param boost_strength: boost strength (0.0 implies no boosting)
    :param boost_strength_factor: Boost strength factor to use [0..1]
    :param duty_cycle_period: The period used to calculate duty cycles
    :param kwinner_local: Whether or not to choose the k-winners locally (across
                          the channels at each location) or globally (across the
                          whole input and across all channels).
    """

    def __init__(self,
                 cnn_out_channels=(64, 64),
                 cnn_percent_on=(0.095, 0.125),
                 cnn_weight_sparsity=(0.5, 0.2),
                 linear_units=1000,
                 linear_percent_on=0.1,
                 linear_weight_sparsity=0.1,
                 boost_strength=1.5,
                 boost_strength_factor=0.9,
                 k_inference_factor=1.0,
                 duty_cycle_period=1000,
                 kwinner_local=False):
        super(GSCSparseCNN, self).__init__()
        # input_shape = (1, 32, 32)
        # First Sparse CNN layer
        if cnn_weight_sparsity[0] < 1.0:
            self.add_module("cnn1", SparseWeights2d(
                nn.Conv2d(1, cnn_out_channels[0], 5),
                weight_sparsity=cnn_weight_sparsity[0]))
        else:
            self.add_module("cnn1", nn.Conv2d(1, cnn_out_channels[0], 5))
        self.add_module("cnn1_batchnorm", nn.BatchNorm2d(cnn_out_channels[0],
                                                         affine=False))
        self.add_module("cnn1_kwinner", KWinners2d(
            channels=cnn_out_channels[0],
            percent_on=cnn_percent_on[0],
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
            local=kwinner_local,
        ))
        self.add_module("cnn1_maxpool", nn.MaxPool2d(2))

        # Second Sparse CNN layer
        if cnn_weight_sparsity[1] < 1.0:
            self.add_module("cnn2", SparseWeights2d(
                nn.Conv2d(cnn_out_channels[0], cnn_out_channels[1], 5),
                weight_sparsity=cnn_weight_sparsity[1]))
        else:
            self.add_module("cnn2", nn.Conv2d(cnn_out_channels[0],
                                              cnn_out_channels[1], 5))
        self.add_module("cnn2_batchnorm",
                        nn.BatchNorm2d(cnn_out_channels[1], affine=False))
        self.add_module("cnn2_kwinner", KWinners2d(
            channels=cnn_out_channels[1],
            percent_on=cnn_percent_on[1],
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
            local=kwinner_local,
        ))
        self.add_module("cnn2_maxpool", nn.MaxPool2d(2))

        self.add_module("flatten", Flatten())

        # Sparse Linear layer
        self.add_module("linear", SparseWeights(
            nn.Linear(25 * cnn_out_channels[1], linear_units),
            weight_sparsity=linear_weight_sparsity))
        self.add_module("linear_bn", nn.BatchNorm1d(linear_units, affine=False))
        self.add_module("linear_kwinner", KWinners(
            n=linear_units,
            percent_on=linear_percent_on,
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period))

        # Classifier
        self.add_module("output", nn.Linear(linear_units, 12))
        self.add_module("softmax", nn.LogSoftmax(dim=1))


class GSCSuperSparseCNN(GSCSparseCNN):
    """Super Sparse CNN model used to classify `Google Speech Commands`
    dataset as described in `How Can We Be So Dense?`_ paper.
    This model provides a sparser version of :class:`GSCSparseCNN`

    .. _`How Can We Be So Dense?`: https://arxiv.org/abs/1903.11257

    """

    def __init__(self):
        super(GSCSuperSparseCNN, self).__init__(
            linear_units=1500,
            linear_weight_sparsity=0.05,
        )


MODEL_URLS = {
    "gsc_sparse_cnn": "http://public.numenta.com/pytorch/hub/gsc_sparse_cnn-eac5f79f.pth",  # noqa: E501
    "gsc_super_sparse_cnn": "http://public.numenta.com/pytorch/hub/gsc_super_sparse_cnn-b1e1921c.pth",  # noqa: E501
}


def gsc_sparse_cnn(pretrained=False, progress=True, **kwargs):
    """
    Sparse CNN model used to classify 'Google Speech Commands' dataset

    :param pretrained: If True, returns a model pre-trained on Google Speech Commands
    :param progress: If True, displays a progress bar of the download to stderr
    :param kwargs: See :class:`GSCSparseCNN`
    """
    model = GSCSparseCNN(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(MODEL_URLS["gsc_sparse_cnn"],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def gsc_super_sparse_cnn(pretrained=False, progress=True):
    """
    Super Sparse CNN model used to classify `Google Speech Commands`
    dataset as described in `How Can We Be So Dense?`_ paper.
    This model provides a sparser version of :class:`GSCSparseCNN`

    :param pretrained: If True, returns a model pre-trained on Google Speech Commands
    :param progress: If True, displays a progress bar of the download to stderr
    """
    model = GSCSuperSparseCNN()
    if pretrained:
        state_dict = load_state_dict_from_url(MODEL_URLS["gsc_super_sparse_cnn"],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
