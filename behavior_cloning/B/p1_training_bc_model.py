"""
Run this to train the inverse model
"""

import torch
import logging
import torch.nn as nn
from torch.utils import data
from learner import BehaviorCloningModel
import numpy as np
import matplotlib.pyplot as plt;
import torch.nn.functional as f

plt.style.use('fivethirtyeight')

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

########### HYPERPARAMETERS ############
NUM_STATE_DIMS = 9
NUM_ACTION_DIMS = 2
HIDDEN_LAYER_SIZES = [32, 16]

BATCH_SIZE = 512
NUM_EPOCHS = 50
LEARNING_RATE = 3e-4
CRITERION = nn.MSELoss()
SEED = 0
ACTIVATION = f.selu

# RESULTS_PATH = './results/'
##########################################


# ##### HYPERPARAMETERS ######
# start_state_dims = 2
# next_state_dims = 2
# action_dims = 4
# nn_layer_1_size = 64
# nn_layer_2_size = 32
# criterion = nn.MSELoss()
# lr = 8e-4
# seed = 0
#
# num_epochs= 140
# bsize = 512
# ############################


def main():
    # Process expert data
    logger.info("Importing data")
    dataset = np.load('./expert.npz')
    tensor_dataset = data.TensorDataset(torch.Tensor(dataset['obs']), torch.Tensor(dataset['action']))
    train_size = int(0.8 * len(tensor_dataset))
    test_size = len(tensor_dataset) - train_size
    train_dataset, test_dataset = data.random_split(tensor_dataset, [train_size, test_size])

    train_loader = data.DataLoader(train_dataset, batch_size=50, shuffle=True)
    valid_loader = data.DataLoader(test_dataset, batch_size=50, shuffle=True)

    logger.info("Importing behavior cloning model")
    model = BehaviorCloningModel(num_envstate_dims=NUM_STATE_DIMS,
                                 num_action_dims=NUM_ACTION_DIMS,
                                 hidden_layer_sizes=HIDDEN_LAYER_SIZES,
                                 criterion=CRITERION,
                                 lr=LEARNING_RATE,
                                 activation=ACTIVATION,
                                 seed=SEED)

    logger.info("Beginning training")
    loss_list, avg_loss_list, valid_loss_list = model.train_and_validate(train_loader, valid_loader, NUM_EPOCHS)

    logger.info(f'Final train loss: {avg_loss_list[-1]}')
    logger.info(f'Final test loss: {valid_loss_list[-1]}')

    # Save trained model
    logger.info("Saving model parameters to bcmodel file")
    torch.save(model.state_dict(), "bcmodel_learned_params.pt")

    # plt.plot(loss_list[1000:])
    # plt.title("Loss")
    # plt.show()

    plt.plot(avg_loss_list, label="Average training loss per epoch")
    plt.plot(valid_loss_list, label="Average validation loss per epoch")
    plt.title("Results over all epochs")
    plt.xlabel("# of Epochs")
    plt.legend()
    plt.show()

    plt.plot(avg_loss_list[5:], label="Average training loss per epoch")
    plt.plot(valid_loss_list[5:], label="Average validation loss per epoch")
    shift = 10
    spacing = 5
    xpos = np.linspace(0, NUM_EPOCHS - shift, int((NUM_EPOCHS - shift) // spacing + 1))
    my_xticks = np.linspace(shift, NUM_EPOCHS, NUM_EPOCHS // spacing)
    my_xticks = [int(i) for i in my_xticks]
    plt.xticks(xpos, my_xticks)
    plt.title(f"Zoomed-In Results (over all but first {shift} epochs)")
    plt.xlabel("# of Epochs")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
