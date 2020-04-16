"""
Run this to train the forward model
"""
import torch
import logging
import torch.nn as nn
from dataset import ObjPushDataset
from model_learners import ForwardModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('fivethirtyeight')

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

##### HYPERPARAMETERS ######
start_state_dims = 2
next_state_dims = 2
action_dims = 4
nn_layer_1_size = 64
nn_layer_2_size = 32
criterion = nn.MSELoss()
lr = 6e-4
seed = 0

num_epochs = 230
bsize = 512
############################


def main():

    train_dir = 'push_dataset/train'
    test_dir = 'push_dataset/test'

    logger.info("Importing data")
    train_loader = DataLoader(ObjPushDataset(train_dir), batch_size=bsize, shuffle=True)
    valid_loader = DataLoader(ObjPushDataset(test_dir), batch_size=bsize, shuffle=True)

    logger.info("Importing forward model")
    model = ForwardModel(start_state_dims=start_state_dims,
                         next_state_dims=next_state_dims,
                         action_dims=action_dims,
                         latent_var_1=nn_layer_1_size,
                         latent_var_2=nn_layer_2_size,
                         criterion=criterion,
                         lr=lr,
                         seed=seed)

    logger.info("Beginning training")
    loss_list, avg_loss_list, valid_loss_list = model.train_and_validate(train_loader, valid_loader, num_epochs)

    logger.info(f'Final train loss: {avg_loss_list[-1]}')
    logger.info(f'Final test loss: {valid_loss_list[-1]}')

    # Save trained model
    logger.info("Saving model parameters to fwdmodel file")
    torch.save(model.state_dict(), "fwdmodel_learned_params.pt")

    # plt.plot(loss_list[1000:])
    # plt.title("Loss")
    # plt.show()

    plt.plot(avg_loss_list, label="Average loss per epoch")
    plt.plot(valid_loss_list, label="Average validation loss per epoch")
    plt.title("Results over all epochs")
    plt.xlabel("# of Epochs")
    plt.legend()
    plt.show()

    plt.plot(avg_loss_list[5:], label="Average training loss per epoch")
    plt.plot(valid_loss_list[5:], label="Average validation loss per epoch")
    shift = 5
    spacing = 5
    xpos = np.linspace(0, num_epochs - shift, int((num_epochs - shift) // spacing + 1))
    my_xticks = np.linspace(shift, num_epochs, num_epochs // spacing)
    my_xticks = [int(i) for i in my_xticks]
    plt.xticks(xpos, my_xticks)
    plt.title(f"Zoomed-In Results (over all but first {shift} epochs)")
    plt.xlabel("# of Epochs")
    plt.legend()
    plt.show()


if __name__=='__main__':
    main()
