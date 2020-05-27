import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from a_c import DEVICE

# Set up logging
import logging
logger = logging.getLogger(__name__)


class BC_Model:
    def __init__(self, policy, batch_size=64, num_epochs=80, learning_rate=1e-4, seed=0):
        torch.manual_seed(seed)
        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.training_loss_list = []
        self.avg_loss_list = []

    def get_loss(self, states, actions):
        # We want a policy that maximizes the expectation of the log probability of drawing a given s,
        # so we use its negative as our loss
        log_probs = self.policy.get_distribution(states).log_prob(actions).float().to(DEVICE)
        bc_loss = -torch.mean(log_probs)
        return bc_loss

    def train(self, expert_data):
        target = len(expert_data) / self.batch_size
        target_int = int(np.ceil(target))
        steps_per_epoch = target_int

        # Train for given number of epochs
        for i in range(self.num_epochs):
            for states, actions in DataLoader(expert_data, batch_size=self.batch_size, shuffle=True):
                states = states.to(DEVICE)
                actions = actions.to(DEVICE)

                # Now we need to update the actor
                # To do so, we first need to get the loss using the function we designed above
                loss = self.get_loss(states=states, actions=actions)

                # Then, we perform the standard gradient update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Finally, for bookkeeping, we append the loss to our training_loss_list
                self.training_loss_list.append(loss.item())

            # Display progress for tracking purposes
            logger.info(f"Completed {i+1}/{self.num_epochs} epochs")

            # Calculate and display mean loss
            mean_loss = np.mean(self.training_loss_list[-steps_per_epoch:])
            logger.info(f"Mean loss: {'{0:.3f}'.format(mean_loss)}")
            self.avg_loss_list.append(mean_loss)
