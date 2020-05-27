import pybullet
from pusher_goal import PusherEnv
import numpy as np
import pandas as pd
import logging
# import ffmpeg

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def main():

    env = PusherEnv(render=True)

    # Ground truth push videos
    logger.info("Recording ground truth videos")
    ground_truth_data_path = "results/P1/true_pushes.csv"

    for i, push in pd.read_csv(ground_truth_data_path, index_col=0).iterrows():
        logger.info(f'Video {i}')

        ################
        # state = push["state"]
        # actions = push["action"]

        state = np.array(push["state"])
        action = np.array([push["d_x"], push["d_y"]])

        # state = np.array([push["obj_x"], push["obj_y"]])
        # actions = [np.array([push["start_push_x"], push["start_push_y"], push["end_push_x"], push["end_push_y"]])]
        ######################

        # Record video
        pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, f"results/P1/vids/true_pushes{i}.mp4")
        env.reset()
        # for action in actions:
        #     state, _, _, _ = env.step(action=action)

        state, _, _, _ = env.step(action=action)

        pybullet.stopStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4)

    # Predicted push videos
    predicted_data_path = "results/P1/pred_pushes.csv"
    logger.info("Recording prediction videos")
    for i, push in pd.read_csv(predicted_data_path, index_col=0).iterrows():
        logger.info(f'Video {i}')

        #######################
        # state = push["state"]
        # actions = push["action"]

        state = np.array(push["state"])
        action = np.array([push["d_x"], push["d_y"]])

        # state = np.array([push["obj_x"], push["obj_y"]])
        # actions = [np.array([push["start_push_x"], push["start_push_y"], push["end_push_x"], push["end_push_y"]])]
        ########################

        # Record video
        pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, f"results/P1/vids/pred_pushes{i}.mp4")
        env.reset()
        # for action in actions:
        #     state, _, _, _ = env.step(action=action)
        state, _, _, _ = env.step(action=action)
        pybullet.stopStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4)


if __name__ == '__main__':
    main()
