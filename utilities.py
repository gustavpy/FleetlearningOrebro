"""Utility functions."""
from collections import OrderedDict
import time
from typing import List, Tuple

from models import Net
import numpy as np
import torch


def train(
    net: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    valloader: torch.utils.data.DataLoader,
    epochs: int = 3,
) -> Tuple[list, list]:
    """Train function.

    Args:
        net (torch.nn.Module): Model
        trainloader (torch.utils.data.DataLoader): train loader
        valloader (torch.utils.data.DataLoader): validaiton loader
        epochs (int, optional): _description_. number of epochs

    Returns:
        tuple[list, list]: _description_
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    net.train()
    opt = torch.optim.Adam(net.model_parameters(), lr=0.001)
    epoch_train_losses = []
    epoch_val_losses = []
    for epoch in range(1, epochs + 1):
        tstart = time.time()
        batch_train_losses = []
        for data_, target_ in trainloader:
            data, target = data_.to(device), target_.to(device)
            opt.zero_grad()
            output = net(data)
            loss = net.loss_fn(output, target)
            loss.backward()
            opt.step()

            batch_train_losses.append(loss.item())
        epoch_train_losses.append(sum(batch_train_losses) / len(batch_train_losses))
        epoch_val_losses.append(test(net, valloader))
      
    return epoch_train_losses, epoch_val_losses


def test(net: Net, testloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
    """Evaluate the network on the entire test set."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    criterion = net.loss_fn
    net.eval()
    loss = []
    with torch.no_grad():
        for images_, labels_ in testloader:
            images, labels = images_.to(device), labels_.to(device)
            outputs = net(images)
            outputs = outputs.unsqueeze(0) if outputs.shape[0] != 1 else outputs
            labels = labels.unsqueeze(0) if labels.shape[0] != 1 else labels
            loss.append(criterion(outputs, labels).item())

    loss = np.mean(loss)
    accuracy = None
    return loss, accuracy


def get_parameters(net: torch.nn.Module) -> List[np.ndarray]:
    """Get model parameters.

    Args:
        net (torch.nn.Module): ML model

    Returns:
        List[np.ndarray]: model parameters
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    # print("⤺ Get model parameters")
    net = net.to("cpu")
    params = [val.numpy() for _, val in net.state_dict().items()]
    net = net.to(device)
    return params


def set_parameters(net: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """Update model parameters."""
    # print("Set model parameters")
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {
            k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
            for k, v in params_dict
        }
    )
    net.load_state_dict(state_dict, strict=True)


def net_instance(name: str) -> torch.nn.Module:
    """Create new model."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    net = Net().to(device)
    print(f"🌻 Created new model - {name} 🌻")
    return net