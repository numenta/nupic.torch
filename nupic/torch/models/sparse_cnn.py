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

from nupic.torch.modules import Flatten, KWinners, KWinners2d, SparseWeights


class MNISTSparseCNN(nn.Sequential):
    """Sparse CNN model used to classify `MNIST` dataset as described in `How
    Can We Be So Dense?`_ paper.

    .. _`How Can We Be So Dense?`: https://arxiv.org/abs/1903.11257

    :param cnn_out_channels: output channels for each CNN layer
    :param cnn_percent_on: Percent of units allowed to remain on each convolution
                           layer
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
    """

    def __init__(self,
                 cnn_out_channels=(32, 64),
                 cnn_percent_on=(0.087, 0.293),
                 linear_units=700,
                 linear_percent_on=0.143,
                 linear_weight_sparsity=0.3,
                 boost_strength=1.5,
                 boost_strength_factor=0.85,
                 k_inference_factor=1.5,
                 duty_cycle_period=1000
                 ):
        super(MNISTSparseCNN, self).__init__(OrderedDict([
            # First Sparse CNN layer
            ("cnn1", nn.Conv2d(1, cnn_out_channels[0], 5)),
            ("cnn1_maxpool", nn.MaxPool2d(2)),
            ("cnn1_kwinner", KWinners2d(channels=cnn_out_channels[0],
                                        percent_on=cnn_percent_on[0],
                                        k_inference_factor=k_inference_factor,
                                        boost_strength=boost_strength,
                                        boost_strength_factor=boost_strength_factor,
                                        duty_cycle_period=duty_cycle_period)),

            # Second Sparse CNN layer
            ("cnn2", nn.Conv2d(cnn_out_channels[0], cnn_out_channels[1], 5)),
            ("cnn2_maxpool", nn.MaxPool2d(2)),
            ("cnn2_kwinner", KWinners2d(channels=cnn_out_channels[1],
                                        percent_on=cnn_percent_on[1],
                                        k_inference_factor=k_inference_factor,
                                        boost_strength=boost_strength,
                                        boost_strength_factor=boost_strength_factor,
                                        duty_cycle_period=duty_cycle_period)),

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
            ("softmax", nn.LogSoftmax())
        ]))


class GSCSparseCNN(nn.Sequential):
    """Sparse CNN model used to classify `Google Speech Commands` dataset as
    described in `How Can We Be So Dense?`_ paper.

    .. _`How Can We Be So Dense?`: https://arxiv.org/abs/1903.11257

    :param cnn_out_channels: output channels for each CNN layer
    :param cnn_percent_on: Percent of units allowed to remain on each convolution
                           layer
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
    """

    def __init__(self,
                 cnn_out_channels=(64, 64),
                 cnn_percent_on=(0.095, 0.125),
                 linear_units=1000,
                 linear_percent_on=0.1,
                 linear_weight_sparsity=0.4,
                 boost_strength=1.5,
                 boost_strength_factor=0.9,
                 k_inference_factor=1.5,
                 duty_cycle_period=1000
                 ):
        super(GSCSparseCNN, self).__init__(OrderedDict([
            # First Sparse CNN layer
            ("cnn1", nn.Conv2d(1, cnn_out_channels[0], 5)),
            ("cnn1_batchnorm", nn.BatchNorm2d(cnn_out_channels[0])),
            ("cnn1_maxpool", nn.MaxPool2d(2)),
            ("cnn1_kwinner", KWinners2d(channels=cnn_out_channels[0],
                                        percent_on=cnn_percent_on[0],
                                        k_inference_factor=k_inference_factor,
                                        boost_strength=boost_strength,
                                        boost_strength_factor=boost_strength_factor,
                                        duty_cycle_period=duty_cycle_period)),

            # Second Sparse CNN layer
            ("cnn2", nn.Conv2d(cnn_out_channels[0], cnn_out_channels[1], 5)),
            ("cnn2_batchnorm", nn.BatchNorm2d(cnn_out_channels[1])),
            ("cnn2_maxpool", nn.MaxPool2d(2)),
            ("cnn2_kwinner", KWinners2d(channels=cnn_out_channels[1],
                                        percent_on=cnn_percent_on[1],
                                        k_inference_factor=k_inference_factor,
                                        boost_strength=boost_strength,
                                        boost_strength_factor=boost_strength_factor,
                                        duty_cycle_period=duty_cycle_period)),

            ("flatten", Flatten()),

            # Sparse Linear layer
            ("linear", SparseWeights(
                nn.Linear(25 * cnn_out_channels[1], linear_units),
                weight_sparsity=linear_weight_sparsity)),
            ("linear_kwinner", KWinners(n=linear_units,
                                        percent_on=linear_percent_on,
                                        k_inference_factor=k_inference_factor,
                                        boost_strength=boost_strength,
                                        boost_strength_factor=boost_strength_factor,
                                        duty_cycle_period=duty_cycle_period)),

            # Classifier
            ("output", nn.Linear(linear_units, 12)),
            ("softmax", nn.LogSoftmax())
        ]))
