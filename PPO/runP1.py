"""
To run the code for each problem, simply run the 'runP#.py' file.
So for this problem, run this file: runP1.py

The P#classes.py files are very similar across problems,
but each includes a scaling which is (roughly) optimized for that specific problem.

The runP#.py file will automatically import the necessary classes from the appropriate location.


[To produce the video submitted you will have to increase 'action_repeat' in the reacher file)
"""


import torch
import torch.nn as nn
from torch.distributions import Normal
import cv2
from reacher import ReacherEnv
import numpy as np
import matplotlib.pyplot as plt
from P1classes import ActorCritic
from P1classes import Memory
from P1classes import PPO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    ############## Hyperparamters ####################
    environment = ReacherEnv()
    state_dim = environment.observation_space.shape[0]
    action_dim = 2

    vid_threshold = 100000

    max_episodes = 40  # max training episodes
    max_timesteps = 190  # max timesteps in one episode
    max_batches = 16
    n_latent_var_a1 = 64
    n_latent_var_a2 = 32
    n_latent_var_c1 = 64
    n_latent_var_c2 = 32
    lr = 1e-4
    gamma = 0.99
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.1  # clip prameter for PPO
    random_seed = 7
    entropy_beta = 0.05
    critic_coef = 1

    std_scale = 0.4

    trial_num = 1.001

    ###################################################

    if random_seed:
        torch.manual_seed(random_seed)

    memory = Memory()
    ppo = PPO(environment=environment, state_dim=state_dim, action_dim=action_dim,
              n_latent_var_a1=n_latent_var_a1, n_latent_var_a2=n_latent_var_a2,
              n_latent_var_c1=n_latent_var_c1, n_latent_var_c2=n_latent_var_c2,
              lr=lr, gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip,
              entropy_beta=entropy_beta, critic_coef=critic_coef)

    # logging variables
    running_reward = 0
    mean_rewards = []
    running_avgs = []

    # Film 'BEFORE'' stochastic video
    ppo.policy_old.film_stochastic_vid(filepath='./pytorch_vids_continuous/p1CONT_{}_BEFORE_stoch_randseed{}.mp4',
                                       trial_num=trial_num, random_seed=random_seed, environment=environment,
                                       max_timesteps=max_timesteps, ppo=ppo, memory=memory, std_scale=std_scale)

    # Film 'BEFORE' deterministic video
    ppo.policy_old.film_deterministic_vid(filepath='./pytorch_vids_continuous/p1CONT_{}_BEFORE_det_randseed{}.mp4',
                                          trial_num=trial_num, random_seed=random_seed, environment=environment,
                                          max_timesteps=max_timesteps, ppo=ppo, memory=memory, std_scale=std_scale)

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = environment.reset()

        for batch in range(max_batches):

            for t in range(max_timesteps):
                # Running policy_old:
                action = ppo.policy_old.act(state, std_scale, memory)
                state, reward, done, _ = environment.step(action)

                # Saving reward and is_terminal:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                running_reward += reward

        episode_mean_reward = running_reward / (max_timesteps * max_batches)

        if episode_mean_reward > vid_threshold:
            # Plot rewards
            plt.plot(range(len(mean_rewards)), mean_rewards, label='Mean Reward per Episode')
            plt.ylabel("Mean Reward")
            plt.xlabel("Number of Episodes")
            plt.title("Performance")
            plt.legend()
            plt.savefig(
                './pytorch_vids_continuous/p1CONT_BEST_{}_randseed{}_performancegraph.png'.format(trial_num, random_seed))
            plt.close()

            plt.plot(range(len(running_avgs)), running_avgs, label='Mean Reward Averaged Over 3 Episodes')
            plt.ylabel("Mean Reward Averaged Over 3 Episodes ")
            plt.xlabel("Number of Episodes")
            plt.title("Averaged Performance")
            plt.legend()
            plt.savefig('./pytorch_vids_continuous/p1CONT_BEST_{}_randseed{}_smoothedperformancegraph.png'.format(trial_num,
                                                                                                             random_seed))
            plt.close()
            ppo.policy_old.film_stochastic_vid(filepath='./pytorch_vids_continuous/p1CONT_{}_BEST_stoch_randseed{}.mp4',
                                               trial_num=trial_num, random_seed=random_seed, environment=environment,
                                               max_timesteps=max_timesteps, ppo=ppo, memory=memory,
                                               std_scale=std_scale)

            ppo.policy_old.film_deterministic_vid(filepath='./pytorch_vids_continuous/p1CONT_{}_BEST_det_randseed{}.mp4',
                                                  trial_num=trial_num, random_seed=random_seed, environment=environment,
                                                  max_timesteps=max_timesteps, ppo=ppo, memory=memory,
                                                  std_scale=std_scale)

        ppo.update(memory, std_scale=std_scale)
        memory.clear_memory()

        # logging
        print("Episode:", i_episode)
        print("Episode mean reward:", episode_mean_reward)

        # Save episode's mean reward and reset running reward
        mean_rewards.append(episode_mean_reward)
        running_reward = 0

        if i_episode > 2:
            target = np.mean(mean_rewards[-3:])
            running_avgs.append(target)

        if i_episode % 23 == 0:
            ppo.policy_old.film_stochastic_vid(filepath='./pytorch_vids_continuous/p1CONT_{}_DURING_stoch_randseed{}.mp4',
                                               trial_num=trial_num, random_seed=random_seed, environment=environment,
                                               max_timesteps=max_timesteps, ppo=ppo, memory=memory,
                                               std_scale=std_scale)

    # Plot rewwards
    plt.plot(range(len(mean_rewards)), mean_rewards, label='Mean Reward per Episode')
    plt.ylabel("Mean Reward")
    plt.xlabel("Number of Episodes")
    plt.title("Performance")
    plt.legend()
    plt.savefig('./pytorch_vids_continuous/p1CONT_{}_randseed{}_performancegraph.png'.format(trial_num, random_seed))
    plt.close()

    plt.plot(range(len(running_avgs)), running_avgs, label='Mean Reward Averaged Over 3 Episodes')
    plt.ylabel("Mean Reward Averaged Over 3 Episodes ")
    plt.xlabel("Number of Episodes")
    plt.title("Averaged Performance")
    plt.legend()
    plt.savefig('./pytorch_vids_continuous/p1CONT_{}_randseed{}_smoothedperformancegraph.png'.format(trial_num, random_seed))
    plt.close()

    # Film 'AFTER'' stochastic video
    ppo.policy_old.film_stochastic_vid(filepath='./pytorch_vids_continuous/p1CONT_{}_AFTER_stoch_randseed{}.mp4',
                                       trial_num=trial_num, random_seed=random_seed, environment=environment,
                                       max_timesteps=max_timesteps, ppo=ppo, memory=memory, std_scale=std_scale)

    # Film 'AFTER' deterministic video
    ppo.policy_old.film_deterministic_vid(filepath='./pytorch_vids_continuous/p1CONT_{}_AFTER_det_randseed{}.mp4',
                                          trial_num=trial_num, random_seed=random_seed, environment=environment,
                                          max_timesteps=max_timesteps, ppo=ppo, memory=memory, std_scale=std_scale)


if __name__ == '__main__':
    main()

