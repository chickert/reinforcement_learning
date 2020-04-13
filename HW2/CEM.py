import torch
import logging
import numpy as np
from sampler import Sampler

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


class CEM:
    """
    UPDATE ALL NAMES????
    """
    def __init__(self, fwd_model, n_iterations, population_size, elite_frac, sigma, alpha):
        self.fwd_model = fwd_model
        self.n_iterations = n_iterations
        self.population_size = population_size
        self.elite_frac = elite_frac
        self.num_elites = int(population_size * elite_frac)
        self.sigma = sigma
        self.alpha = alpha
        self.sampler = Sampler(sigma=sigma, population_size=population_size)

    def evaluate_action(self, action, start_state, goal_state):
        pred_state = self.fwd_model(start_state, action)
        """
        CHECK THIS
        """
        return torch.norm(pred_state - goal_state, 2).item() ** 2

    def update_sampler_params(self, push_angles, push_lengths, losses):
        """
        DO THESE NAMES MAKE SENSE
        """
        elites = np.argsort(losses)[:self.num_elites]
        self.sampler.push_ang_mean = self.alpha * np.mean(push_angles[elites]) + \
                                       (1 - self.alpha) * self.sampler.push_ang_mean
        self.sampler.push_len_mean = self.alpha * np.mean(push_lengths[elites]) + \
                                        (1 - self.alpha) * self.sampler.push_len_mean

    def plan_action(self, start_state, goal_state):
        for i in range(self.n_iterations):
            push_angles, push_lengths, actions = self.sampler.sample_population(start_state=start_state)
            losses = np.array([self.evaluate_action(action=action, start_state=start_state, goal_state=goal_state) \
                               for action in actions])
            self.update_sampler_params(push_angles=push_angles, push_lengths=push_lengths, losses=losses)
        """
        DO I WANT LOGGER STUFF HERE
        """
        planned_action = self.sampler.get_argmax_action(start_state=start_state)
        self.sampler.reset()
        return planned_action
