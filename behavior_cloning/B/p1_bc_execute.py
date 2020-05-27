"""
After training with p1_training_bc_model.py, run this for p1
Then run p1_make_vids.py to generate the videos
"""

import torch
import logging
import torch.nn as nn

from learner import BehaviorCloningModel
from torch.utils import data

from pusher_goal import PusherEnv
import numpy as np
import pandas as pd
import torch.nn.functional as f

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

NUM_PUSHES = 10

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
# num_epochs = 140
# bsize = 512
#
# num_pushes = 10
#
# ############################


def main():
    logger.info("Instantiating model and importing weights")
    # instantiate model and import pretrained weights
    bc_model = BehaviorCloningModel(num_envstate_dims=NUM_STATE_DIMS,
                                    num_action_dims=NUM_ACTION_DIMS,
                                    hidden_layer_sizes=HIDDEN_LAYER_SIZES,
                                    criterion=CRITERION,
                                    lr=LEARNING_RATE,
                                    activation=ACTIVATION,
                                    seed=SEED)

    bc_model.load_state_dict(torch.load("bcmodel_learned_params.pt"))

    # Load in data
    logger.info("Importing test data")
    dataset = np.load('./expert.npz')
    tensor_dataset = data.TensorDataset(torch.Tensor(dataset['obs']), torch.Tensor(dataset['action']))
    train_size = int(0.7 * len(tensor_dataset))
    test_size = len(tensor_dataset) - train_size
    train_dataset, test_dataset = data.random_split(tensor_dataset, [train_size, test_size])

    # only want 1 push each time, so set batch_size to 1
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    env = PusherEnv()

    errors = []
    true_pushes = []
    pred_pushes = []

    logger.info("Running loop")
    for i, (state, action) in enumerate(test_loader):
        logger.info(f'Iteration #{i}')
        # Convert inputs to floats
        state = state.float()
        action = action.float()

        # Use model to predict action given state
        pred_action = bc_model(state)

        # Switch output from tensors to numpy for easy use later
        state = state.data.numpy()[0]
        action = action.data.numpy()[0]
        pred_action = pred_action.data.numpy()[0]


        end_state, _, _, _ = env.step(action=pred_action)
        end_state = np.array(end_state)

        # Calculate errors
        action_error = np.linalg.norm(action - pred_action)
        state_error = np.linalg.norm(state - end_state)

        # Keep the results

        ###################
        errors.append(dict(action_error=action_error, state_error=state_error))
        true_pushes.append(dict(d_x=action[0], d_y=action[1], state=state))
        pred_pushes.append(dict(d_x=action[0], d_y=action[1], state=end_state))

        # true_pushes.append(dict(obj_x=start_state[0], obj_y=start_state[1], start_push_x=true_action[0],
        #                         start_push_y=true_action[1], end_push_x=true_action[2], end_push_y=true_action[3]))
        # pred_pushes.append(dict(obj_x=start_state[0], obj_y=start_state[1], start_push_x=pred_action[0],
        #                         start_push_y=pred_action[1], end_push_x=pred_action[2], end_push_y=pred_action[3]))
        ###################

        if i > NUM_PUSHES - 1:
            break

        pd.DataFrame(errors).to_csv("results/P1/bc_model_errors.csv")
        pd.DataFrame(true_pushes).to_csv("results/P1/true_pushes.csv")
        pd.DataFrame(pred_pushes).to_csv("results/P1/pred_pushes.csv")


if __name__ == '__main__':
    main()
