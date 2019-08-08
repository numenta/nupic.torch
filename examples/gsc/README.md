# Google Speech Commands example

In this example we run a sparse convolutional neural network on the
[Google Speech Commands](https://arxiv.org/abs/1804.03209) spoken
digit dataset, as in the paper
[How Can We Be So Dense]()https://arxiv.org/abs/1903.11257. During
testing, we add white noise to the dataset and measure how well the
network classifies the noisy data.

## Usage

These Python scripts download and prepare the Google Speech Commands
dataset, then they run the nupic.torch `GSCSparseCNN` or
`GSCSuperSparseCNN` on the dataset.

### Option 1: Inference only

Dataset download size: 127 MB

```
pip install git+https://github.com/numenta/nupic.torch.git#egg=nupic.torch
pip install -r requirements.txt
python download_preprocessed_data.py
python run_gsc_model.py --pretrained
```

### Option 2: Learning and inference

Dataset download size: 1.5 GB

```
pip install git+https://github.com/numenta/nupic.torch.git#egg=nupic.torch
pip install -r requirements.txt
python download_raw_data.py
python run_gsc_model.py
```

### Model configuration

Two optional flags

```
python run_gsc_model.py --pretrained --supersparse
```

- `--pretrained`: Use pretrained model rather than training locally
- `--supersparse`: Use "super sparse" network rather than basic sparse network
