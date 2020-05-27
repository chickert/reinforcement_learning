import logging

from torch.utils.data import DataLoader

from learner import BehaviorCloningModel
from training import train
import numpy as np
import torch
from torch.utils import data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


####### PARAMS ############
NUM_STATE_DIMS = 9
NUM_ACTION_DIMS = 2
HIDDEN_LAYER_SIZES = [32, 16]

BATCH_SIZE = 512
NUM_EPOCHS = 50
LEARNING_RATE = 3e-4

RESULTS_PATH = './results/'
###########################


def main():

    # Process expert data
    logger.info("Importing data")
    dataset = np.load('./expert.npz')
    tensor_dataset = data.TensorDataset(torch.Tensor(dataset['obs']), torch.Tensor(dataset['action']))
    train_size = int(0.8 * len(tensor_dataset))
    test_size = len(tensor_dataset) - train_size
    train_dataset, test_dataset = data.random_split(tensor_dataset, [train_size, test_size])

    train_loader = data.DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=50, shuffle=True)
    ####

    logger.info("Importing behavior cloning model")
    model = BehaviorCloningModel(num_envstate_dims=NUM_STATE_DIMS,
                                 num_action_dims=NUM_ACTION_DIMS,
                                 hidden_layer_sizes=HIDDEN_LAYER_SIZES)

    logger.info("Training model...")
    logger.info("-" * 50)
    train(model=model,
          train_data=train_data,
          learning_rate=LEARNING_RATE,
          num_epochs=NUM_EPOCHS,
          model_path=f'{RESULTS_PATH}/bc_model_params',
          training_loss_path=f'{RESULTS_PATH}/bc_model_training_loss')

    # Likely need to change this or add in a log_metrics thing
    log_metrics(model=model,
                train_data=train_data,
                test_data=test_data)


if __name__ == "__main__":
    main()
