import argparse
import glob
import re

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from nupic.torch.models.sparse_cnn import gsc_sparse_cnn, gsc_super_sparse_cnn
from nupic.torch.modules import rezero_weights, update_boost_strength


LEARNING_RATE = 0.01
LEARNING_RATE_GAMMA = 0.9
MOMENTUM = 0.0
EPOCHS = 2  # 15
FIRST_EPOCH_BATCH_SIZE = 4
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 1000
TEST_BATCH_SIZE = 1000
REDUCE_LR_ON_PLATEAU = False


def train(model, loader, optimizer, criterion, device):
    """
    Train the model using given dataset loader.
    Called on every epoch.

    :param model: pytorch model to be trained
    :type model: torch.nn.Module
    :param loader: DataLoader configured for the epoch.
    :type loader: :class:`torch.utils.data.DataLoader`
    :param optimizer: Optimizer object used to train the model.
    :type optimizer: :class:`torch.optim.Optimizer`
    :param criterion: loss function to use
    :type criterion: function
    :param device:
    :type device: :class:`torch.device`
    """
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(loader, desc="Train",
                                                    leave=False)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test(model, loader, criterion, device, desc="Test"):
    """
    Evaluate trained model using given dataset loader.
    Called on every epoch.

    :param model: Pretrained pytorch model
    :type model: torch.nn.Module
    :param loader: DataLoader configured for the epoch.
    :type loader: :class:`torch.utils.data.DataLoader`
    :param criterion: loss function to use
    :type criterion: function
    :param device:
    :type device: :class:`torch.device`
    :param desc: Description for progress bar
    :type desc: str
    :return: Dict with "accuracy", "loss" and "total_correct"
    """
    model.eval()
    loss = 0
    total_correct = 0
    with torch.no_grad():
        for data, target in tqdm(loader, desc=desc, leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            loss += criterion(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()

    return {"accuracy": total_correct / len(loader.dataset),
            "loss": loss / len(loader.dataset),
            "total_correct": total_correct}


def run_example(supersparse, pretrained):
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For this example we will use the default values.
    # See GSCSparseCNN documentation for all possible parameters and their values.
    modelclass = (gsc_super_sparse_cnn if supersparse else gsc_sparse_cnn)

    model = modelclass(pretrained=pretrained).to(device)
    print("Model:")
    print(model)

    if not pretrained:
        # Train
        x_train, y_train = map(torch.tensor, np.load("data/gsc_train.npz").values())
        x_valid, y_valid = map(torch.tensor, np.load("data/gsc_valid.npz").values())
        x_test, y_test = map(torch.tensor, np.load("data/gsc_test.npz").values())

        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        valid_dataset = torch.utils.data.TensorDataset(x_valid, y_valid)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

        first_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=FIRST_EPOCH_BATCH_SIZE, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)

        sgd = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        lr_scheduler = optim.lr_scheduler.StepLR(sgd, step_size=1,
                                                 gamma=LEARNING_RATE_GAMMA)

        for epoch in range(EPOCHS):
            loader = (first_loader if epoch == 0 else train_loader)
            train(model=model, loader=loader, optimizer=sgd,
                  criterion=F.nll_loss, device=device)
            if REDUCE_LR_ON_PLATEAU:
                validation = test(model=model, loader=valid_loader,
                                  criterion=F.nll_loss, device=device,
                                  desc="Validation")
                lr_scheduler.step(validation["loss"])
            else:
                lr_scheduler.step()
            model.apply(rezero_weights)
            model.apply(update_boost_strength)

            results = test(model=model, loader=test_loader, criterion=F.nll_loss,
                           device=device)
            print("Epoch {}: {}".format(epoch, results))

    # Test on the noisy data.
    for filepath in sorted(glob.glob("data/gsc_test_noise*.npz")):
        suffix = re.match("data/gsc_test_noise(.*).npz", filepath).groups()[0]
        noise = float("0.{}".format(suffix))
        x_test, y_test = map(torch.tensor, np.load(filepath).values())
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)
        results = test(model=model, loader=test_loader, criterion=F.nll_loss,
                       device=device)

        print("Noise level: {}, Results: {}".format(noise, results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supersparse", action="store_true")
    parser.add_argument("--pretrained", action="store_true")

    args = parser.parse_args()
    run_example(supersparse=args.supersparse, pretrained=args.pretrained)
