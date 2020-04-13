import torch
import numpy as np
from push_env import PushingEnv


class Sampler:
    def __init__(self, sigma, population_size):
        self.environment = PushingEnv
        self.sigma = sigma
        self.population_size = population_size
        self.push_ang_mean = 0
        self.push_ang_std = np.pi * self.sigma
        self.push_len_mean = self.environment.push_len_min + 0.5 * self.environment.push_len_range
        self.push_len_std = 0.5 * self.environment.push_len_range * self.sigma
        self.push_len_min = self.environment.push_len_min
        self.push_len_max = self.environment.push_len_min + self.environment.push_len_range

    def sample_push_angle(self):
        while True:
            push_ang = self.push_ang_mean + self.push_ang_std * np.random.randn()
            if -np.pi < push_ang < np.pi:
                break
        return push_ang

    def sample_push_length(self):
        while True:
            push_len = self.push_len_mean + self.push_len_std * np.random.randn()
            if self.push_len_min < push_len < self.push_len_max:
                break
        return push_len

    def sample(self, start_state):
        while True:
            push_ang = self.sample_push_angle()
            push_len = self.sample_push_length()

            # Use angle and length to calculate the action (initial and final positions of robot’s arm tip)
            obj_x, obj_y = start_state.data.np()[0]
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
        return push_ang, push_len, torch.from_numpy(np.array(action)).float().unsqueeze(0)

    def sample_population(self, start_state):
        push_angles, push_lengths, actions = zip(*[self.sample(start_state) for _ in range(self.population_size)])
        return np.array(push_angles), np.array(push_lengths), actions

    def get_argmax_action(self, start_state):
        # Use angle and length to calculate the action (initial and final positions of robot’s arm tip)
        obj_x, obj_y = start_state.data.np()[0]
        start_x = obj_x - self.push_len_min * np.cos(self.push_ang_mean)
        start_y = obj_y - self.push_len_min * np.sin(self.push_ang_mean)
        end_x = obj_x + self.push_len_mean * np.cos(self.push_ang_mean)
        end_y = obj_y + self.push_len_mean * np.sin(self.push_ang_mean)
        action = start_x, start_y, end_x, end_y
        return np.array(action)

    def reset(self):
        self.push_ang_mean = 0
        self.push_len_mean = self.environment.push_len_min + 0.5 * self.environment.push_len_range

