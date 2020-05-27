import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


class InverseModel(nn.Module):
    """
    Inverse model predicts action given the current state and the desired future state
    """

    def __init__(self, start_state_dims, next_state_dims, action_dims,
                 latent_var_1=64, latent_var_2=32, criterion=nn.MSELoss(), lr=4e-4, seed=0):
        torch.manual_seed(seed)
        super(InverseModel, self).__init__()
        self.state_dims = start_state_dims + next_state_dims
        self.model = nn.Sequential(
            nn.Linear(self.state_dims, latent_var_1),
            nn.ReLU(),
            nn.Linear(latent_var_1, latent_var_2),
            nn.ReLU(),
            nn.Linear(latent_var_2, action_dims)
        )
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, combined_states):
        actions = self.model(combined_states)
        return actions

    def train_and_validate(self, train_dl, valid_dl, num_epochs):
        loss_list = []
        avg_loss_list = []
        valid_loss_list = []
        logger.info("Starting with epoch 0")
        for epoch in range(num_epochs):
            losses_for_given_epoch = []
            self.model.train()
            for start_states, next_states, true_actions in train_dl:

                start_states = start_states.float()
                next_states = next_states.float()
                true_actions = true_actions.float()

                self.optimizer.zero_grad()
                combined_states = torch.cat((start_states, next_states), dim=1)
                pred_actions = self.model(combined_states)
                loss = self.criterion(pred_actions, true_actions)

                loss.backward()
                self.optimizer.step()
                losses_for_given_epoch.append(loss.item())

            self.model.eval()
            with torch.no_grad():
                valid_loss_sum = 0
                for start_states, next_states, true_actions in valid_dl:

                    start_states = start_states.float()
                    next_states = next_states.float()
                    true_actions = true_actions.float()

                    combined_states = torch.cat((start_states, next_states), dim=1)
                    pred_actions = self.model(combined_states)
                    valid_loss_sum += self.criterion(pred_actions, true_actions)

                valid_loss = valid_loss_sum / len(valid_dl)

            loss_list += losses_for_given_epoch
            avg_loss_list.append(np.mean(losses_for_given_epoch))
            valid_loss_list.append(valid_loss)
            logger.info(f'Completed epoch: {epoch}/{num_epochs}')
            logger.info(f'Avg loss this epoch: {np.mean(losses_for_given_epoch)}')
            logger.info(f'Validation loss this epoch: {valid_loss}')

        return loss_list, avg_loss_list, valid_loss_list


class ForwardModel(nn.Module):
    """
    Forward model predicts future state given current state and action
    """
    def __init__(self, start_state_dims, next_state_dims, action_dims,
                 latent_var_1=64, latent_var_2=32, criterion=nn.MSELoss(), lr=4e-4, seed=0):
        torch.manual_seed(seed)
        super(ForwardModel, self).__init__()
        self.state_dims = start_state_dims + action_dims
        self.model = nn.Sequential(
            nn.Linear(self.state_dims, latent_var_1),
            nn.ReLU(),
            nn.Linear(latent_var_1, latent_var_2),
            nn.ReLU(),
            nn.Linear(latent_var_2, next_state_dims)
        )
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, combined_input):
        next_states = self.model(combined_input)
        return next_states

    def train_and_validate(self, train_dl, valid_dl, num_epochs):
        loss_list = []
        avg_loss_list = []
        valid_loss_list = []
        logger.info("Starting with epoch 0")
        for epoch in range(num_epochs):
            losses_for_given_epoch = []
            self.model.train()
            for start_states, next_states, true_actions in train_dl:

                start_states = start_states.float()
                next_states = next_states.float()
                true_actions = true_actions.float()

                self.optimizer.zero_grad()
                combined_input = torch.cat((start_states, true_actions), dim=1)
                pred_states = self.model(combined_input)
                loss = self.criterion(pred_states, next_states)

                loss.backward()
                self.optimizer.step()
                losses_for_given_epoch.append(loss.item())

            self.model.eval()
            with torch.no_grad():
                valid_loss_sum = 0
                for start_states, next_states, true_actions in valid_dl:

                    start_states = start_states.float()
                    next_states = next_states.float()
                    true_actions = true_actions.float()

                    combined_input = torch.cat((start_states, true_actions), dim=1)
                    pred_states = self.model(combined_input)
                    valid_loss_sum += self.criterion(pred_states, next_states)

                valid_loss = valid_loss_sum / len(valid_dl)

            loss_list += losses_for_given_epoch
            avg_loss_list.append(np.mean(losses_for_given_epoch))
            valid_loss_list.append(valid_loss)
            logger.info(f'Completed epoch: {epoch}/{num_epochs}')
            logger.info(f'Avg loss this epoch: {np.mean(losses_for_given_epoch)}')
            logger.info(f'Validation loss this epoch: {valid_loss}')

        return loss_list, avg_loss_list, valid_loss_list

