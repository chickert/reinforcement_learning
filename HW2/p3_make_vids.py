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

    # Ground truth push videos (inv)
    logger.info("Recording inverse ground truth videos")
    inv_ground_truth_data_path = "results/P3/inv_ground_truth_pushes.csv"

    for i, push in pd.read_csv(inv_ground_truth_data_path, index_col=0).iterrows():
        logger.info(f'Video {i}')
        state = np.array([push["obj_x"], push["obj_y"]])
        actions = [
            np.array([push["start_push_x_1"], push["start_push_y_1"], push["end_push_x_1"], push["end_push_y_1"]]),
            np.array([push["start_push_x_2"], push["start_push_y_2"], push["end_push_x_2"], push["end_push_y_2"]])
        ]
        # Record video
        pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, f"results/P3/vids/inv_ground_truth_pushes{i}.mp4")
        env.reset_box(pos=[state[0], state[1], env.box_z])
        for action in actions:
            _, state = env.execute_push(*action)
        pybullet.stopStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4)

    # Predicted push videos (inv)
    inv_predicted_data_path = "results/P3/inv_predicted_pushes.csv"
    logger.info("Recording inverse prediction videos")
    for i, push in pd.read_csv(inv_predicted_data_path, index_col=0).iterrows():
        logger.info(f'Video {i}')
        state = np.array([push["obj_x"], push["obj_y"]])
        actions = [
            np.array([push["start_push_x_1"], push["start_push_y_1"], push["end_push_x_1"], push["end_push_y_1"]]),
            np.array([push["start_push_x_2"], push["start_push_y_2"], push["end_push_x_2"], push["end_push_y_2"]])
        ]
        # Record video
        pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, f"results/P3/vids/inv_pred_pushes{i}.mp4")
        env.reset_box(pos=[state[0], state[1], env.box_z])
        for action in actions:
            _, state = env.execute_push(*action)
        pybullet.stopStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4)

    # Ground truth push videos (Forward)
    logger.info("Recording fwd ground truth videos")
    fwd_ground_truth_data_path = "results/P3/fwd_ground_truth_pushes.csv"
    for i, push in pd.read_csv(fwd_ground_truth_data_path, index_col=0).iterrows():
        logger.info(f'Video {i}')
        state = np.array([push["obj_x"], push["obj_y"]])
        actions = [
            np.array([push["start_push_x_1"], push["start_push_y_1"], push["end_push_x_1"], push["end_push_y_1"]]),
            np.array([push["start_push_x_2"], push["start_push_y_2"], push["end_push_x_2"], push["end_push_y_2"]])
        ]
        # Record video
        pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, f"results/P3/vids/fwd_ground_truth_pushes{i}.mp4")
        env.reset_box(pos=[state[0], state[1], env.box_z])
        for action in actions:
            _, state = env.execute_push(*action)
        pybullet.stopStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4)

    # Predicted push videos (forward)
    fwd_predicted_data_path = "results/P3/fwd_predicted_pushes.csv"
    logger.info("Recording fwd prediction videos")
    for i, push in pd.read_csv(fwd_predicted_data_path, index_col=0).iterrows():
        logger.info(f'Video {i}')
        state = np.array([push["obj_x"], push["obj_y"]])
        actions = [
            np.array([push["start_push_x_1"], push["start_push_y_1"], push["end_push_x_1"], push["end_push_y_1"]]),
            np.array([push["start_push_x_2"], push["start_push_y_2"], push["end_push_x_2"], push["end_push_y_2"]])
        ]
        # Record video
        pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, f"results/P3/vids/fwd_pred_pushes{i}.mp4")
        env.reset_box(pos=[state[0], state[1], env.box_z])
        for action in actions:
            _, state = env.execute_push(*action)
        pybullet.stopStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4)


if __name__ == '__main__':
    main()
