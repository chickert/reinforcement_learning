"""
Run this file for Problem 1
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

num_pushes_in_vid = 10
vid_path = './results/p1/p1_video_behavioralcloning.mp4'
num_episodes_to_evaluate_on = 100

batch_size = 128
num_epochs = 70
learning_rate = 2e-4

act_layer_one = 128
act_layer_two = 64
crit_layer_one = 64
crit_layer_two = 32
actor_std = 4e-2

def main():

    # Load data
    expert_data = np.load("./expert.npz")
    expert_data = TensorDataset(torch.tensor(expert_data["obs"]), torch.tensor(expert_data["action"]))

    # Instantiate the environment (had to modify it slightly from the form given to make for easier recording later)
    environment = PusherEnvModified()

    policy = ActorCritic(state_space_dimension=environment.state_space_dimension,
                         action_space_dimension=environment.action_space_dimension,
                         actor_hidden_layer_units=(act_layer_one, act_layer_two),
                         critic_hidden_layer_units=(crit_layer_one, crit_layer_two), actor_std=4e-2,
                         activation=nn.Tanh)

    # Use the policy from above to instantiate our behavioral cloning model
    bc_model = BC_Model(policy=deepcopy(policy), batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate)

    # Train model and save resulting policy parameters
    bc_model.train(expert_data=expert_data)
    bc_model.policy.save(path="./results/p1/bc_model_params.pt")
    pd.DataFrame(bc_model.training_loss_list, columns=["train_loss"]).to_csv("./results/p1/bc_train_loss.csv")
    pd.DataFrame(bc_model.avg_loss_list, columns=["avg_train_loss"]).to_csv("./results/p1/bc_avg_train_loss.csv")

    # Plot training loss
    plt.plot(bc_model.training_loss_list, label="Training loss")
    plt.title("Loss as a Function of Time")
    plt.xlabel("# of batches")
    plt.legend()
    plt.savefig("./results/p1/bc_train_loss_chart.png")
    plt.close()

    # Plot avg. training loss
    plt.plot(bc_model.avg_loss_list, label="Average training loss per epoch")
    plt.title("Avg. Loss as a Function of Time")
    plt.xlabel("# of epochs")
    plt.legend()
    plt.savefig("./results/p1/bc_avg_train_loss_chart.png")
    plt.close()

    # Now use the policy from the post-training behavioral cloning model, and compare the results
    produced_model = PPO_Model(environment=environment, policy=deepcopy(bc_model.policy), n_steps_per_trajectory=64)

    # For comparison, we evaluate the learned policy on 100 episodes
    ltwo_dist_list = []
    trajectories_list = []
    for i in range(num_episodes_to_evaluate_on):
        _, actions, _, _ = produced_model.generate_trajectory(use_argmax=True, perform_reset=False)

        trajectories_list.append(actions)
        state = produced_model.environment.simulator.get_obs()
        ltwo_dist_list.append(np.linalg.norm(state[3:6] - state[6:9]))
        produced_model.environment.reset()

    pd.DataFrame({"mean_L2_distance": np.mean(ltwo_dist_list),
                  "standard_L2dist": np.std(ltwo_dist_list) / np.sqrt(len(ltwo_dist_list))},
                 index=["BC"]).to_csv("./results/p1/bc_l2distance.csv")


    # Using the trajectories generated above,
    # make video showing evaluation of policy on 10 episodes
    env_for_vid = PusherEnv(render=True)
    env_for_vid.render()
    vid_output = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    for given_trajectory in trajectories_list[:num_pushes_in_vid]:
        for action in given_trajectory:

            # apply action and record into video
            env_for_vid.apply_action(action)
            scene_image = env_for_vid.robot.cam.get_images(get_rgb=True, get_depth=False)[0]
            vid_output.write(np.array(scene_image))

        # Reset video environment after a given push
        env_for_vid.reset()


if __name__ == "__main__":
    main()