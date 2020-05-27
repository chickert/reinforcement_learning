import torch
import logging
import numpy as np
from sampler import Sampler

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


class CEM:
    def __init__(self, fwd_model, n_iterations, population_size, elite_frac, ang_sigma, len_sigma, smoothing_param):
        self.fwd_model = fwd_model
        self.n_iterations = n_iterations
        self.population_size = population_size
        self.elite_frac = elite_frac
        self.num_elites = int(population_size * elite_frac)
        self.smoothing_param = smoothing_param
        self.sampler = Sampler(ang_sigma=ang_sigma, len_sigma=len_sigma, population_size=population_size)

    def action_plan(self, start_state, goal_state):
        for i in range(self.n_iterations):

            push_angles, push_lengths, actions = self.sampler.sample_whole_population(start_state=start_state)

            # Find losses associated with each action
            losses = []
            for action in actions:
                # Added for loop to handle extrapolation case (P3; P2 should never enter this)
                if start_state.shape == torch.Size([2]):
                    print("Modified state shape for consistency")
                    start_state = torch.unsqueeze(start_state, 0)
                combined_input = torch.cat((start_state, action), dim=1)
                pred_state = self.fwd_model(combined_input)
                loss = torch.norm(pred_state - goal_state).item()
                losses.append(loss)

            # Sort out elites from the losses
            elites_indices = np.argsort(losses)[:self.num_elites]
            # print("PUSH ANG MEAN:", self.sampler.push_ang_mean)
            # print("PUSH LEN MEAN:", self.sampler.push_len_mean)

            # Perform update of sampler params:
            self.sampler.push_ang_mean = self.smoothing_param * np.mean(push_angles[elites_indices]) + \
                                         (1 - self.smoothing_param) * self.sampler.push_ang_mean
            self.sampler.push_len_mean = self.smoothing_param * np.mean(push_lengths[elites_indices]) + \
                                         (1 - self.smoothing_param) * self.sampler.push_len_mean

        planned_action = self.sampler.get_best_action(start_state=start_state)
        self.sampler.reset()
        return planned_action
