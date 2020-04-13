import torch
import torch.nn as nn
from CEM import CEM
import logging
from model_learners import ForwardModel
from dataset import ObjPushDataset
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

##### HYPERPARAMETERS ######
"""
DO THESE NAMES MAKE SENSE??
"""
start_state_dims = 2
next_state_dims = 2
action_dims = 4
nn_layer_1_size = 64
nn_layer_2_size = 32
criterion = nn.MSELoss()
lr = 3e-4
seed = 0

"""
Do these defaults make sense?
"""

n_iterations = 50
population_size = 2048
elite_frac = 0.2
sigma = 1e-2
alpha = 0.98

num_pushes = 10


############################

def main():
    """
    Do names of everything make sense?
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

    # instantiate CEM for planning
    cem = CEM(fwd_model=fwd_model,
              n_iterations=n_iterations,
              population_size=population_size,
              elite_frac=elite_frac,
              sigma=sigma,
              alpha=alpha)

    # Load in data
    logger.info("Importing test data")
    test_dir = 'push_dataset/test'
    # only want 1 push each time, so set batch_size to 1
    test_loader = DataLoader(ObjPushDataset(test_dir), batch_size=1, shuffle=True)

    true_actions = []
    pred_actions = []

    for i, (start_state, goal_state, true_action) in enumerate(test_loader):
        # Convert inputs to floats
        start_state = start_state.float()
        goal_state = goal_state.float()
        true_action = true_action.float()

        # Generate planned action and compare to true action
        planned_action = cem.plan_action(start_state=start_state, goal_state=goal_state)

        """
        Is this the right distance to use?
        """
        torch.norm(true_action - planned_action, 2).item() ** 2

        if i > num_pushes - 1:
            break


if __name__ == '__main__':
    main()
