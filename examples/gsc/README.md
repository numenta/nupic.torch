# Google Speech Commands example

In this example we run a sparse convolutional neural network on the
[Google Speech Commands](https://arxiv.org/abs/1804.03209) spoken
digit dataset, as in the paper
[How Can We Be So Dense](https://arxiv.org/abs/1903.11257). During
testing, we add white noise to the dataset and measure how well the
network classifies the noisy data.

## Usage

These Python scripts download and prepare the Google Speech Commands
dataset, then they run the nupic.torch `GSCSparseCNN` or
`GSCSuperSparseCNN` on the dataset.

### Install nupic.torch and requirements

```
pip install git+https://github.com/numenta/nupic.torch.git#egg=nupic.torch
pip install -r requirements.txt
```

### Option 1: Download preprocessed input data

Dataset download size: 1.8 GB

```
python download_preprocessed_data.py
python run_gsc_model.py
```

### Option 2: Download wav files, process them on your machine

Dataset download size: 1.5 GB

```
python download_raw_data.py
python run_gsc_model.py
```

### Model configuration

```
python run_gsc_model.py --pretrained --supersparse --seed 42
```

- `--pretrained`: Use pretrained model rather than training locally
- `--supersparse`: Use "super sparse" network rather than basic sparse network
- `--seed`: Random seed
