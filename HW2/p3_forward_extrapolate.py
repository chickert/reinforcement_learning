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


def fwd_infer(cem):
    def infer(start_state, goal_state):
        start_state = torch.tensor(start_state).float()
        goal_state = torch.tensor(goal_state).float()
        return cem.action_plan(start_state, goal_state)
    return infer


def project_ahead(start_state, goal_state, infer, env):
    env.reset_box()
    actions = []
    for _ in range(2):
        action = infer(start_state, goal_state)
        actions.append(action)
        _, start_state = env.execute_push(*action)
    env.reset_box()
    return start_state, actions


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
    infer = fwd_infer(cem)

    errors = []
    true_pushes = []
    pred_pushes = []

    for i in range(num_pushes):
        # Sample push
        start_state, goal_state, (action_1, action_2) = cem.sampler.environment.plan_model_extrapolate(seed=i)
        start_state = torch.from_numpy(start_state).unsqueeze(0)
        goal_state = torch.from_numpy(goal_state).unsqueeze(0)
        start_state = start_state.float()
        goal_state = goal_state.float()
        final_state, (predicted_action_1, predicted_action_2) = project_ahead(start_state, goal_state, infer, env=cem.sampler.environment)

        start_state = start_state[0]
        goal_state = goal_state[0]
        start_state = start_state.numpy()

        # Calculate errors
        action_1_error = np.linalg.norm(action_1 - predicted_action_1)
        action_2_error = np.linalg.norm(action_2 - predicted_action_2)
        state_error = np.linalg.norm(goal_state - final_state)

        # Keep the results
        errors.append(dict(action_1_error=action_1_error, action_2_error=action_2_error, state_error=state_error))

        true_pushes.append(dict(obj_x=start_state[0], obj_y=start_state[1], start_push_x_1=action_1[0],
                                start_push_y_1=action_1[1], end_push_x_1=action_1[2], end_push_y_1=action_1[3],
                                start_push_x_2=action_2[0], start_push_y_2=action_2[1], end_push_x_2=action_2[2],
                                end_push_y_2=action_2[3]))

        pred_pushes.append(dict(obj_x=start_state[0], obj_y=start_state[1], start_push_x_1=predicted_action_1[0],
                                start_push_y_1=predicted_action_1[1], end_push_x_1=predicted_action_1[2],
                                end_push_y_1=predicted_action_1[3], start_push_x_2=predicted_action_2[0],
                                start_push_y_2=predicted_action_2[1], end_push_x_2=predicted_action_2[2],
                                end_push_y_2=predicted_action_2[3]))

    logger.info("Saving output to csv files")
    pd.DataFrame(errors).to_csv("results/P3/forward_model_extrap_errors.csv")
    pd.DataFrame(true_pushes).to_csv("results/P3/fwd_true_pushes.csv")
    pd.DataFrame(pred_pushes).to_csv("results/P3/fwd_pred_pushes.csv")


if __name__ == '__main__':
    main()