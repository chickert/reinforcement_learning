import pybullet
from push_env import PushingEnv
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def main():
    env = PushingEnv(ifRender=True)

    # Ground truth push videos
    logger.info("Recording ground truth videos")
    ground_truth_data_path = "results/P1/ground_truth_pushes.csv"

    for i, push in pd.read_csv(ground_truth_data_path, index_col=0).iterrows():
        logger.info(f'Video {i}')
        state = np.array([push["obj_x"], push["obj_y"]])
        actions = [np.array([push["start_push_x"], push["start_push_y"], push["end_push_x"], push["end_push_y"]])]

        # Record video
        pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, f"results/P1/vids/ground_truth_pushes{i}.mp4")
        env.reset_box(pos=[state[0], state[1], env.box_z])
        for action in actions:
            _, state = env.execute_push(*action)
        pybullet.stopStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4)

    # Predicted push videos
    predicted_data_path = "results/P1/predicted_pushes.csv"
    logger.info("Recording prediction videos")
    for i, push in pd.read_csv(predicted_data_path, index_col=0).iterrows():
        logger.info(f'Video {i}')
        state = np.array([push["obj_x"], push["obj_y"]])
        actions = [np.array([push["start_push_x"], push["start_push_y"], push["end_push_x"], push["end_push_y"]])]

        # Record video
        pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, f"results/P1/vids/pred_pushes{i}.mp4")
        env.reset_box(pos=[state[0], state[1], env.box_z])
        for action in actions:
            _, state = env.execute_push(*action)
        pybullet.stopStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4)


if __name__ == '__main__':
    main()
