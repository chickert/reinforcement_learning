import gym
import torch
from q_network import Qnetwork
from agent import Agent
from replay_memory import SARSD, ReplayMemory
import ffmpeg

##### HYPERPARAMETERS #####
LAYER_1_NODES = 512
LAYER_2_NODES = 256
LR = 1e-4
NUM_EPISODES = 6
MODEL_LOAD_PATH = './trained_model'
RANDOM_VIDEO_PATH = './random_model_vid'
TRAINED_VIDEO_PATH = './trained_model_vid'
T_or_F = True
##########################


def main(trained_model=False):
    env = gym.make('CartPole-v0')
    if not trained_model:
        env = gym.wrappers.Monitor(env, RANDOM_VIDEO_PATH, video_callable=lambda episode_id: True, force=True)
    if trained_model:
        env = gym.wrappers.Monitor(env, TRAINED_VIDEO_PATH, video_callable=lambda episode_id: True, force=True)

    q_network = Qnetwork(
        observation_shape=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        layer_1_nodes=LAYER_1_NODES,
        layer_2_nodes=LAYER_2_NODES,
        lr=LR
    )
    q_network.load_state_dict(torch.load(MODEL_LOAD_PATH))

    for i_episode in range(NUM_EPISODES):
        observation = env.reset()
        for t in range(700):
            env.render()
            if not trained_model:
                action = env.action_space.sample()
            if trained_model:
                observation = torch.Tensor(observation)
                q_vals = q_network(observation)
                action = q_vals.argmax().item()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()


if __name__ == '__main__':
    main(trained_model=T_or_F)