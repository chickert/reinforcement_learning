"""
To run the code for each problem, simply run the 'runP#.py' file.
So for this problem, run runP3.py

The P#classes.py files are very similar across problems,
but each includes a scaling which is (roughly) optimized for that specific problem.

The runP#.py file will automatically import the necessary classes from the appropriate location.

"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import cv2
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var_a1, n_latent_var_a2, n_latent_var_c1, n_latent_var_c2):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var_a1),
            nn.ReLU(),
            nn.Linear(n_latent_var_a1, n_latent_var_a2),
            nn.ReLU(),
            nn.Linear(n_latent_var_a2, action_dim)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var_c1),
            nn.ReLU(),
            nn.Linear(n_latent_var_c1, n_latent_var_c2),
            nn.ReLU(),
            nn.Linear(n_latent_var_c2, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, std_scale, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Normal(loc=action_probs, scale=std_scale)
        action = dist.sample()

        action = 1 * action

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.detach().numpy()


    def act_deterministic(self, state, std_scale, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Normal(loc=action_probs, scale=std_scale)
        action = action_probs
        action = 1 * action

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.detach().numpy()

    def evaluate(self, state, action, std_scale):
        action_probs = self.action_layer(state)
        dist = Normal(loc=action_probs, scale=std_scale)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def film_stochastic_vid(self, filepath, trial_num, random_seed, environment, max_timesteps, ppo, memory,
                            std_scale):
        out = cv2.VideoWriter(filepath.format(trial_num, random_seed),
                              cv2.VideoWriter_fourcc(*'mp4v'), 30,
                              (640, 480))
        img = environment.render()
        out.write(np.array(img))
        state = environment.reset()
        for scene in range(max_timesteps):
            action = ppo.policy_old.act(state, std_scale, memory)
            next_state, reward, done, _ = environment.step(action)
            img = environment.render()
            out.write(np.array(img))
            state = next_state
        out.release()
        memory.clear_memory()

    def film_deterministic_vid(self, filepath, trial_num, random_seed, environment, max_timesteps, ppo, memory,
                               std_scale):
        out = cv2.VideoWriter(filepath.format(trial_num, random_seed),
                              cv2.VideoWriter_fourcc(*'mp4v'), 30,
                              (640, 480))
        img = environment.render()
        out.write(np.array(img))
        state = environment.reset()
        for scene in range(max_timesteps):
            action = ppo.policy_old.act_deterministic(state, std_scale, memory)
            next_state, reward, done, _ = environment.step(action)
            img = environment.render()
            out.write(np.array(img))
            state = next_state
        out.release()
        memory.clear_memory()


class PPO:
    def __init__(self, environment, state_dim, action_dim, n_latent_var_a1, n_latent_var_a2, n_latent_var_c1,
                 n_latent_var_c2, lr, gamma, K_epochs, eps_clip, entropy_beta, critic_coef):
        self.environment = environment
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_beta = entropy_beta
        self.critic_coef = critic_coef

        self.policy = ActorCritic(state_dim, action_dim,
                                  n_latent_var_a1, n_latent_var_a2,
                                  n_latent_var_c1, n_latent_var_c2).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim,
                                      n_latent_var_a1, n_latent_var_a2,
                                      n_latent_var_c1, n_latent_var_c2).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory, std_scale):

        # I found that for my implementation, using this form of the rollouts worked best
        disc_reward = 0
        rewards_bin = []

        # We begin with the latest rewards, and work backwards
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                disc_reward = 0

            disc_reward = (disc_reward * self.gamma) + reward

            # Insert backwards, since we 'reversed' above.
            rewards_bin.insert(0, disc_reward)


        rewards = torch.tensor(rewards_bin).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)


        # Must convert lists to tensors
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_states = torch.stack(memory.states).to(device).detach()

        # Now we optimize the policy
        for _ in range(self.K_epochs):

            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, std_scale=std_scale)

            # First we find the ratio of the probabilities of selecting action a_t, given state s_t, under
            # the new and old policies, respectively.
            # We can use the log to make this more computationally efficient.
            newold_ratio = torch.exp(logprobs - old_logprobs.detach())

            # subtract of the state-values from the rewards to get the advantages
            advantages = rewards - state_values.detach()

            # Reshape this
            newold_ratio = newold_ratio.view(2, -1)

            target1 = newold_ratio * advantages

            # In pytorch, 'clamp' is how we clip.
            target2 = torch.clamp(newold_ratio, 1 - self.eps_clip, 1 + self.eps_clip)
            target3 = target2 * advantages

            # We need to isolate out the third term to reshape it appropriately
            entropy = self.entropy_beta * dist_entropy
            entropy = entropy.view(2, -1)
            actor_loss = -torch.min(target1, target3)
            critic_loss = self.critic_coef * self.MseLoss(state_values, rewards)

            # Now we have our total loss
            loss = actor_loss + critic_loss - entropy

            # now perform update via gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())