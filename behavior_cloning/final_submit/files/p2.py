"""
Run this file for Problem 2
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import logging
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.style.use('seaborn-darkgrid')
from copy import deepcopy
from pusher_goal import PusherEnv
from pusher_mod import PusherEnvModified
from bc_model import BC_Model
from ppo_model import PPO_Model
from a_c import ActorCritic

logger = logging.basicConfig(level=logging.INFO)

num_episodes_to_evaluate_on = 100
num_pushes_in_vid = 10

act_layer_one = 128
act_layer_two = 64
crit_layer_one = 64
crit_layer_two = 32
actor_std=4e-2

n_steps_per_trajectory = 64
n_trajectories_per_batch = 16
n_epochs = 5

n_iterations = 100
learning_rate = 2e-4
clipping_param = 0.2
entropy_coefficient = 1e-3

train_critic_only_on_init = True
seed = 0


def main():
    # Load data
    expert_data = np.load("./expert.npz")
    expert_data = TensorDataset(torch.tensor(expert_data["obs"]), torch.tensor(expert_data["action"]))

    # Instantiate the environment (had to modify it slightly from the form given to make for easier recording later)
    environment = PusherEnvModified()

    # Instantiate the three models according to the problem statement
    policy = ActorCritic(state_space_dimension=environment.state_space_dimension,
                         action_space_dimension=environment.action_space_dimension,
                         actor_hidden_layer_units=(act_layer_one, act_layer_two),
                         critic_hidden_layer_units=(crit_layer_one, crit_layer_two), actor_std=actor_std,
                         activation=nn.Tanh)

    fromscratch_model = PPO_Model(environment=environment, policy=deepcopy(policy), bc_coefficient=0,
                                  n_steps_per_trajectory=n_steps_per_trajectory,
                                  n_trajectories_per_batch=n_trajectories_per_batch, n_epochs=n_epochs,
                                  n_iterations=n_iterations, learning_rate=learning_rate,
                                  clipping_param=clipping_param, entropy_coefficient=entropy_coefficient, seed=seed)

    policy.load(path="./results/p1/bc_model_params.pt")
    jointlossfinetune_model = PPO_Model(environment=environment, policy=deepcopy(policy), bc_coefficient=0.1,
                                        n_steps_per_trajectory=n_steps_per_trajectory,
                                        n_trajectories_per_batch=n_trajectories_per_batch, n_epochs=n_epochs,
                                        n_iterations=n_iterations, learning_rate=learning_rate,
                                        clipping_param=clipping_param, entropy_coefficient=entropy_coefficient,
                                        seed=seed)

    vanillafinetune_model = PPO_Model(environment=environment, policy=deepcopy(policy), bc_coefficient=0,
                                      n_steps_per_trajectory=n_steps_per_trajectory,
                                      n_trajectories_per_batch=n_trajectories_per_batch, n_epochs=n_epochs,
                                      n_iterations=n_iterations, learning_rate=learning_rate,
                                      clipping_param=clipping_param, entropy_coefficient=entropy_coefficient,
                                      seed=seed)

    # Train each
    vanillafinetune_model.train(train_critic_only_on_init=train_critic_only_on_init)
    jointlossfinetune_model.train(expert_data=expert_data, train_critic_only_on_init=train_critic_only_on_init)
    fromscratch_model.train()

    # First, generate results and video for model trained from scratch
    fromscratch_model.save_training_rewards("./results/p2/rewards_fromscratchmodel")

    fromscratch_ltwo_dist_list = []
    fromscratch_trajectories_list = []
    for i in range(num_episodes_to_evaluate_on):
        _, actions, _, _ = fromscratch_model.generate_trajectory(use_argmax=True, perform_reset=False)

        fromscratch_trajectories_list.append(actions)
        state = fromscratch_model.environment.simulator.get_obs()
        fromscratch_ltwo_dist_list.append(np.linalg.norm(state[3:6] - state[6:9]))
        fromscratch_model.environment.reset()

    pd.DataFrame({"mean_L2_distance": np.mean(fromscratch_ltwo_dist_list),
                  "standard_L2dist": np.std(fromscratch_ltwo_dist_list) / np.sqrt(len(fromscratch_ltwo_dist_list))},
                 index=["from_scratch"]).to_csv("./results/p2/l2distances_fromscratchmodel.csv")

    # Using the trajectories generated above,
    # make video showing evaluation of policy on 10 episodes
    env_for_vid = PusherEnv(render=True)
    env_for_vid.render()
    vid_output = cv2.VideoWriter("./results/p2/p2_video_fromscratchmodel.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    for given_trajectory in fromscratch_trajectories_list[:num_pushes_in_vid]:
        for action in given_trajectory:

            # apply action and record into video
            env_for_vid.apply_action(action)
            scene_image = env_for_vid.robot.cam.get_images(get_rgb=True, get_depth=False)[0]
            vid_output.write(np.array(scene_image))

        # Reset video environment after a given push
        env_for_vid.reset()

    # Second, generate results and video for joint-loss fine-tuned model
    jointlossfinetune_model.save_training_rewards("./results/p2/rewards_jointlossfinetuned")

    jointlossfinetuned_ltwo_dist_list = []
    jointlossfinetuned_trajectories_list = []
    for i in range(num_episodes_to_evaluate_on):
        _, actions, _, _ = jointlossfinetune_model.generate_trajectory(use_argmax=True, perform_reset=False)

        jointlossfinetuned_trajectories_list.append(actions)
        state = jointlossfinetune_model.environment.simulator.get_obs()
        jointlossfinetuned_ltwo_dist_list.append(np.linalg.norm(state[3:6] - state[6:9]))
        jointlossfinetune_model.environment.reset()

    pd.DataFrame({"mean_L2_distance": np.mean(jointlossfinetuned_ltwo_dist_list),
                  "standard_L2dist": np.std(jointlossfinetuned_ltwo_dist_list) / np.sqrt(len(jointlossfinetuned_ltwo_dist_list))},
                 index=["jointloss_finetuned"]).to_csv("./results/p2/l2distances_jointlossfinetunedhmodel.csv")

    # Using the trajectories generated above,
    # make video showing evaluation of policy on 10 episodes

    vid_output = cv2.VideoWriter("./results/p2/p2_video_jointlossfinetunedmodel.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    for given_trajectory in jointlossfinetuned_trajectories_list[:num_pushes_in_vid]:
        for action in given_trajectory:

            # apply action and record into video
            env_for_vid.apply_action(action)
            scene_image = env_for_vid.robot.cam.get_images(get_rgb=True, get_depth=False)[0]
            vid_output.write(np.array(scene_image))

        # Reset video environment after a given push
        env_for_vid.reset()

    # Third, generate results and video for vanilla fine-tuned model
    vanillafinetune_model.save_training_rewards("./results/p2/rewards_vanillafinetunedhmodel")

    vanillafinetuned_ltwo_dist_list = []
    vanillafinetuned_trajectories_list = []
    for i in range(num_episodes_to_evaluate_on):
        _, actions, _, _ = vanillafinetune_model.generate_trajectory(use_argmax=True, perform_reset=False)

        vanillafinetuned_trajectories_list.append(actions)
        state = vanillafinetune_model.environment.simulator.get_obs()
        vanillafinetuned_ltwo_dist_list.append(np.linalg.norm(state[3:6] - state[6:9]))
        vanillafinetune_model.environment.reset()

    pd.DataFrame({"mean_L2_distance": np.mean(vanillafinetuned_ltwo_dist_list),
                  "standard_L2dist": np.std(vanillafinetuned_ltwo_dist_list) / np.sqrt(len(vanillafinetuned_ltwo_dist_list))},
                 index=["vanilla_finetuned"]).to_csv("./results/p2/l2distances_vanillafinetunedmodel.csv")

    # Using the trajectories generated above,
    # make video showing evaluation of policy on 10 episodes

    vid_output = cv2.VideoWriter("./results/p2/p2_video_vanillafinetunedmodel.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    for given_trajectory in vanillafinetuned_trajectories_list[:num_pushes_in_vid]:
        for action in given_trajectory:

            # apply action and record into video
            env_for_vid.apply_action(action)
            scene_image = env_for_vid.robot.cam.get_images(get_rgb=True, get_depth=False)[0]
            vid_output.write(np.array(scene_image))

        # Reset video environment after a given push
        env_for_vid.reset()

    # Plot the learning curves for each policy
    plt.plot(fromscratch_model.mean_rewards, label="From-scratch policy")
    plt.plot(jointlossfinetune_model.mean_rewards, label="Joint-loss fine-tuned policy")
    plt.plot(vanillafinetune_model.mean_rewards, label='Vanilla fine-tuned policy')
    plt.title("Learning Curves for the Three Policies")
    plt.ylabel("Mean Rewards")
    plt.legend()
    plt.savefig("./results/p2/learningcurves_chart.png")
    plt.close()


if __name__ == "__main__":
    main()
