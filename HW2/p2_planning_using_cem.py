import torch
import torch.nn as nn
from CEM import CEM
import logging
from model_learners import ForwardModel
from dataset import ObjPushDataset
from torch.utils.data import Dataset, DataLoader
from push_env import PushingEnv

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
lr = 1e-4
seed = 0

"""
xx_defaults
xx_names
"""

n_iterations = 200
population_size = 100
elite_frac = 0.2
ang_sigma = 2e-1
len_sigma = 1
smoothing_param = 1     # when set to 1, no smoothing is applied

num_pushes = 10


############################

def main():
    """
    xx_names
    """
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

    # Load in data
    logger.info("Importing test data")
    test_dir = 'push_dataset/test'
    # only want 1 push each time, so set batch_size to 1
    test_loader = DataLoader(ObjPushDataset(test_dir), batch_size=1, shuffle=True)

    true_actions = []
    pred_actions = []

    logger.info("Running loop")
    for i, (start_state, goal_state, true_action) in enumerate(test_loader):
        logger.info(f'Iteration #{i}')
        # Convert inputs to floats
        start_state = start_state.float()
        goal_state = goal_state.float()
        true_action = true_action.float()

        # Generate planned action and compare to true action
        planned_action = cem.action_plan(start_state=start_state, goal_state=goal_state)

        # target = torch.norm(true_action - planned_action)
        # logger.info(f'Distance b/w final object pose (after push) and goal object pose {i}: {target}')

        # Execute planned action
        pusher = PushingEnv()
        start_x, start_y, end_x, end_y = planned_action
        _, output_state = pusher.execute_push(start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y)

        # Compare output state to goal state (should be L2 distance around 0.025)
        diff = torch.norm(goal_state - output_state)
        logger.info(f'Distance b/w final object pose (after push) and goal object pose {i}: {diff}')

        if i > num_pushes - 1:
            break


if __name__ == '__main__':
    main()
