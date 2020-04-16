"""
After training with p2_training_forward_model.py, run this for p2
Then run p2_make_vids.py to generate the videos
"""

import torch
import torch.nn as nn
from CEM import CEM
import logging
from model_learners import ForwardModel
from dataset import ObjPushDataset
from torch.utils.data import Dataset, DataLoader
from push_env import PushingEnv
import pandas as pd
import numpy as np

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

n_iterations = 200
population_size = 100
elite_frac = 0.2
ang_sigma = 2e-1
len_sigma = 1
smoothing_param = 1     # when set to 1, no smoothing is applied

num_pushes = 10

############################

def main():
    logger.info("Instantiating model and importing weights")
    # instantiate forward model and import pretrained weights
    fwd_model = ForwardModel(start_state_dims=start_state_dims,
                             next_state_dims=next_state_dims,
                             action_dims=action_dims,
                             latent_var_1=nn_layer_1_size,
                             latent_var_2=nn_layer_2_size,
                             criterion=criterion,
                             lr=lr,
                             seed=seed)
    fwd_model.load_state_dict(torch.load("fwdmodel_learned_params.pt"))

    logger.info("Instantiating CEM for planning")
    # instantiate CEM for planning
    cem = CEM(fwd_model=fwd_model,
              n_iterations=n_iterations,
              population_size=population_size,
              elite_frac=elite_frac,
              ang_sigma=ang_sigma,
              len_sigma=len_sigma,
              smoothing_param=smoothing_param
              )

    errors = []
    true_pushes = []
    pred_pushes = []
    fails = 0

    # Load in data
    logger.info("Importing test data")
    test_dir = 'push_dataset/test'
    # only want 1 push each time, so set batch_size to 1
    test_loader = DataLoader(ObjPushDataset(test_dir), batch_size=1, shuffle=True)

    logger.info("Running loop")
    for i, (start_state, goal_state, true_action) in enumerate(test_loader):

        logger.info(f'Iteration #{i}')
        # Convert inputs to floats
        start_state = start_state.float()
        goal_state = goal_state.float()
        true_action = true_action.float()

        # Generate planned action and compare to true action
        try:
            planned_action = cem.action_plan(start_state=start_state, goal_state=goal_state)
        except ValueError:
            planned_action = None
            fails += 1

        if planned_action is not None:
            # Switch output from tensors to numpy for easy use later
            start_state = start_state.data.numpy()[0]
            goal_state = goal_state.data.numpy()[0]
            true_action = true_action.data.numpy()[0]

            # Execute planned action
            _, output_state = np.array(cem.sampler.environment.execute_push(*planned_action))

            # Calculate errors
            action_error = np.linalg.norm(true_action - planned_action)
            state_error = np.linalg.norm(goal_state - output_state)

            # Keep the results
            errors.append(dict(action_error=action_error, state_error=state_error))
            true_pushes.append(dict(obj_x=start_state[0], obj_y=start_state[1], start_push_x=true_action[0],
                                    start_push_y=true_action[1], end_push_x=true_action[2], end_push_y=true_action[3]))
            pred_pushes.append(dict(obj_x=start_state[0], obj_y=start_state[1], start_push_x=planned_action[0],
                                    start_push_y=planned_action[1], end_push_x=planned_action[2], end_push_y=planned_action[3]))

        if i - fails > num_pushes - 1:
            break

        pd.DataFrame(errors).to_csv("results/P2/forward_model_errors.csv")
        pd.DataFrame(true_pushes).to_csv("results/P2/true_pushes.csv")
        pd.DataFrame(pred_pushes).to_csv("results/P2/pred_pushes.csv")


if __name__ == '__main__':
    main()
