import torch
import numpy as np
from push_env import PushingEnv
import logging

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


class Sampler:
    def __init__(self, ang_sigma, len_sigma, population_size):
        self.environment = PushingEnv(ifRender=False)
        self.population_size = population_size
        self.push_ang_mean = 0
        self.push_ang_std = np.pi * ang_sigma
        self.push_len_mean = self.environment.push_len_min + 0.5 * self.environment.push_len_range
        self.push_len_std = 0.5 * self.environment.push_len_range * len_sigma
        self.push_len_min = self.environment.push_len_min
        self.push_len_max = self.environment.push_len_min + self.environment.push_len_range

    def sample_whole_population(self, start_state):
        # logger.info("in sample population")
        push_angles, push_lengths, actions = zip(*[self.sample(start_state) for _ in range(self.population_size)])
        # logger.info("converting to np")
        return np.array(push_angles), np.array(push_lengths), actions

    def sample(self, start_state):
        # logger.info("entered sample while loop")
        while True:
            push_ang = self.gen_push_angle()
            push_len = self.gen_push_length()

            # Use angle and length to calculate the action (initial and final positions of robot’s arm tip)
            obj_x, obj_y = start_state.data.numpy()[0]
            start_x = obj_x - self.push_len_min * np.cos(push_ang)
            start_y = obj_y - self.push_len_min * np.sin(push_ang)
            end_x = obj_x + push_len * np.cos(push_ang)
            end_y = obj_y + push_len * np.sin(push_ang)
            action = start_x, start_y, end_x, end_y

            # Assess feasibility of these start and end positions
            start_radius = np.sqrt(start_x**2 + start_y**2)
            end_radius = np.sqrt(end_x**2 + end_y**2)
            # find valid push that does not lock the arm
            # and find push that does not push obj out of workspace (camera view)
            if start_radius < self.environment.max_arm_reach \
                    and end_radius + self.push_len_min < self.environment.max_arm_reach \
                    and self.environment.workspace_min_x < end_x < self.environment.workspace_max_x \
                    and self.environment.workspace_min_y < end_y < self.environment.workspace_max_y:
                break
        # logger.info("exited sample while loop")
        return push_ang, push_len, torch.from_numpy(np.array(action)).float().unsqueeze(0)

    def gen_push_length(self):
        while True:
            push_len = self.push_len_mean + self.push_len_std * np.random.randn()
            if self.push_len_min < push_len < self.push_len_max:
                break
        return push_len

    def gen_push_angle(self):
        while True:
            push_ang = self.push_ang_mean + self.push_ang_std * np.random.randn()
            if -np.pi < push_ang < np.pi:
                break
        return push_ang

    def get_best_action(self, start_state):
        # Use angle and length to calculate the action (initial and final positions of robot’s arm tip)
        obj_x, obj_y = start_state.data.numpy()[0]
        start_x = obj_x - self.push_len_min * np.cos(self.push_ang_mean)
        start_y = obj_y - self.push_len_min * np.sin(self.push_ang_mean)
        end_x = obj_x + self.push_len_mean * np.cos(self.push_ang_mean)
        end_y = obj_y + self.push_len_mean * np.sin(self.push_ang_mean)
        action = start_x, start_y, end_x, end_y
        return np.array(action)

    def reset(self):
        self.push_ang_mean = 0
        self.push_len_mean = self.environment.push_len_min + 0.5 * self.environment.push_len_range

