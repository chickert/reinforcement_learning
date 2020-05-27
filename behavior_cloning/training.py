import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from learner import BehaviorCloningModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)


def train(
        model,
        train_data,
        criterion,
        num_epochs=10,
        learning_rate=1e-3,
        model_path=None,
        training_loss_path=None):

    # Send model to device and initialize optimizer
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    for i in range(num_epochs):

        epoch_losses = []
        for states, actions in train_data:

            states = states.float().to(DEVICE)
            actions = actions.float().to(DEVICE)

            # Generate predictions
            preds = model(states).to(DEVICE)
            loss = criterion(preds, actions)

            # Take gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.cpu().item())

        losses += epoch_losses
        logger.info(f'epochs completed: \t {i + 1}/{num_epochs}')
        logger.info(f'mean loss: \t {"{0:.2E}".format(np.mean(epoch_losses))}')
        logger.info("-" * 50)

    if model_path:
        torch.save(model.state_dict(), f'{model_path}.pt')

    if training_loss_path:
        pd.DataFrame(losses).to_csv(f'{training_loss_path}.csv')