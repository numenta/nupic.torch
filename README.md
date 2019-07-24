## Numenta Platform for Intelligent Computing PyTorch libraries

The Numenta Platform for Intelligent Computing (**NuPIC**) is a machine intelligence platform that implements the [HTM learning algorithms](https://numenta.com/resources/papers-videos-and-more/).  For more information, see [numenta.org](http://numenta.org) or [HTM Forum](https://discourse.numenta.org/c/engineering/machine-learning).

This library integrates selected neuroscience principles from HTM into the [pytorch](https://pytorch.org/) deep learning platform. The current code aims to replicate how sparsity is enforced via Spatial Pooling, as defined in the paper [How Could We Be So Dense? The Benefits of Using Highly Sparse Representations](https://arxiv.org/abs/1903.11257).

### Installation

To install from local source code:
    
    python setup.py develop

### Test

To run all tests:

    python setup.py test

### Examples

We've created a few jupyter notebooks demonstrating how to use **nupic.torch** with standard datasets. You can find these notebooks in the [examples/](https://github.com/numenta/nupic.torch/tree/master/examples/) directory or if you prefer you can open them in [colab](http://colab.research.google.com/github/numenta/nupic.torch/) and start experimenting. 


### _Having problems?_

For any installation issues, please [search our forums](https://discourse.numenta.org/search?q=tag%3Ainstallation%20category%3A10) (post questions there). You can report bugs at https://github.com/numenta/nupic.torch/issues.


