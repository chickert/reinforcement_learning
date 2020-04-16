import torch
import logging
import torch.nn as nn
from dataset import ObjPushDataset
from model_learners import InverseModel
from torch.utils.data import Dataset, DataLoader
from push_env import PushingEnv
import numpy as np
import pandas as pd

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
lr = 8e-4
seed = 0

num_epochs = 140
bsize = 512

num_pushes = 10

############################


def main():
    logger.info("Instantiating model and importing weights")
    # instantiate forward model and import pretrained weights
    inv_model = InverseModel(start_state_dims=start_state_dims,
                             next_state_dims=next_state_dims,
                             action_dims=action_dims,
                             latent_var_1=nn_layer_1_size,
                             latent_var_2=nn_layer_2_size,
                             criterion=criterion,
                             lr=lr,
                             seed=seed)

    inv_model.load_state_dict(torch.load("invmodel_learned_params.pt"))

    # Load in data
    logger.info("Importing test data")
    test_dir = 'push_dataset/test'
    # only want 1 push each time, so set batch_size to 1
    test_loader = DataLoader(ObjPushDataset(test_dir), batch_size=1, shuffle=True)

    env = PushingEnv()

    errors = []
    true_pushes = []
    pred_pushes = []

    logger.info("Running loop")
    for i, (start_state, goal_state, true_action) in enumerate(test_loader):
        logger.info(f'Iteration #{i}')
        # Convert inputs to floats
        start_state = start_state.float()
        goal_state = goal_state.float()
        true_action = true_action.float()

        # Use inverse model to predict action given the start and goal states
        combined_input = torch.cat((start_state, goal_state), dim=1)
        pred_action = inv_model(combined_input)

        # Switch output from tensors to numpy for easy use later
        start_state = start_state.data.numpy()[0]
        goal_state = goal_state.data.numpy()[0]
        true_action = true_action.data.numpy()[0]
        pred_action = pred_action.data.numpy()[0]

        start_x, start_y, end_x, end_y = pred_action
        _, end_state = env.execute_push(start_x, start_y, end_x, end_y)
        end_state = np.array(end_state)

        # Calculate errors
        action_error = np.linalg.norm(true_action - pred_action)
        state_error = np.linalg.norm(goal_state - end_state)

        # Keep the results
        errors.append(dict(action_error=action_error, state_error=state_error))
        true_pushes.append(dict(obj_x=start_state[0], obj_y=start_state[1], start_push_x=true_action[0],
                                start_push_y=true_action[1], end_push_x=true_action[2], end_push_y=true_action[3]))
        pred_pushes.append(dict(obj_x=start_state[0], obj_y=start_state[1], start_push_x=pred_action[0],
                                start_push_y=pred_action[1], end_push_x=pred_action[2], end_push_y=pred_action[3]))

        if i > num_pushes - 1:
            break

        pd.DataFrame(errors).to_csv("results/P1/inverse_model_errors.csv")
        pd.DataFrame(true_pushes).to_csv("results/P1/true_pushes.csv")
        pd.DataFrame(pred_pushes).to_csv("results/P1/pred_pushes.csv")


if __name__ == '__main__':
    main()
