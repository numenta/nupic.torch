# Numenta Platform for Intelligent Computing PyTorch libraries

[![CircleCI](https://circleci.com/gh/numenta/nupic.torch.svg?style=svg)](https://circleci.com/gh/numenta/nupic.torch)

This library integrates selected neuroscience principles from Hierarchical Temporal Memory (HTM) into the [pytorch](https://pytorch.org/) deep learning platform. The current code aims to replicate how sparsity is enforced via Spatial Pooling, as defined in the paper [*How Could We Be So Dense? The Benefits of Using Highly Sparse Representations*](https://arxiv.org/abs/1903.11257).

For detail on the neuroscience behind these theories, read [Why Neurons Have Thousands of Synapses, A Theory of Sequence Memory in Neocortex](https://numenta.com/neuroscience-research/research-publications/papers/why-neurons-have-thousands-of-synapses-theory-of-sequence-memory-in-neocortex/). For a description of _Spatial Pooling_ in isolation, read [*Spatial Pooling (BAMI)*](https://numenta.com/resources/biological-and-machine-intelligence/spatial-pooling-algorithm/).

`nupic.torch` is named after the original HTM library, the [Numenta Platform for Intelligent Computing (*NuPIC*)](https://github.com/numenta/nupic).


Interested in [contributing](CONTRIBUTING.md)?

## Installation

To install from local source code:
    
    python setup.py develop

Or using conda:

    conda env create

### Test

To run all tests:

    python setup.py test

## Examples

We've created a few jupyter notebooks demonstrating how to use **nupic.torch** with standard datasets. You can find these notebooks in the [examples/](https://github.com/numenta/nupic.torch/tree/master/examples/) directory or if you prefer you can open them in [Google Colab](http://colab.research.google.com/github/numenta/nupic.torch/) and start experimenting. 


## _Having problems?_

For any installation issues, please [search our forums](https://discourse.numenta.org/search?q=tag%3Ainstallation%20category%3A10) (post questions there). Report bugs [here](https://github.com/numenta/nupic.torch/issues/new/).
