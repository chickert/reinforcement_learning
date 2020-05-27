import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from mlp import MultiLayerPerceptron
import logging
import numpy as np


logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


class BehaviorCloningModel(nn.Module):
    """
    Model predicts action given state of the environment
    """
    def __init__(self, num_envstate_dims, num_action_dims,
                 hidden_layer_sizes, criterion=nn.MSELoss(), lr=4e-4, activation=f.selu, seed=0):
        torch.manual_seed(seed)
        super(BehaviorCloningModel, self).__init__()
        self.mlp = MultiLayerPerceptron(
            in_features=num_envstate_dims,
            hidden_layer_sizes=hidden_layer_sizes + [num_action_dims],
            activation=activation)
        self.criterion = criterion
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=lr)

    def forward(self, state):
        return self.mlp(state)

    def train_and_validate(self, train_dl, valid_dl, num_epochs):
        loss_list = []
        avg_loss_list = []
        valid_loss_list = []
        logger.info("Starting with epoch 0")
        for epoch in range(num_epochs):
            losses_for_given_epoch = []
            self.mlp.train()
            for states, actions in train_dl:
                states = states.float()
                actions = actions.float()

                self.optimizer.zero_grad()
                # Generate predictions
                pred_actions = self.mlp(states)
                loss = self.criterion(pred_actions, actions)

                loss.backward()
                self.optimizer.step()
                losses_for_given_epoch.append(loss.item())

            self.mlp.eval()
            with torch.no_grad():
                valid_loss_sum = 0
                for states, actions in valid_dl:
                    states = states.float()
                    actions = actions.float()

                    pred_actions = self.mlp(states)
                    valid_loss_sum += self.criterion(pred_actions, actions)

                valid_loss = valid_loss_sum / len(valid_dl)

            loss_list += losses_for_given_epoch
            avg_loss_list.append(np.mean(losses_for_given_epoch))
            valid_loss_list.append(valid_loss)
            logger.info(f'Completed epoch: {epoch}/{num_epochs}')
            logger.info(f'Avg loss this epoch: {np.mean(losses_for_given_epoch)}')
            logger.info(f'Validation loss this epoch: {valid_loss}')

        return loss_list, avg_loss_list, valid_loss_list
