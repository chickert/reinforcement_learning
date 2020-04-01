"""
This file is associated with P2


"""
import time

import numpy as np
from airobot import Robot
from airobot.utils.common import ang_in_mpi_ppi
from airobot.utils.common import clamp
from airobot.utils.common import euler2quat
from airobot.utils.common import quat_multiply
from airobot.utils.common import rotvec2quat
from gym import spaces
import pybullet as p


class ReacherWallEnv:
	def __init__(self, action_repeat=10, render=False): # 10
		self._action_repeat = action_repeat		
		self.robot = Robot('ur5e_stick', pb=True, pb_cfg={'gui': render, 'realtime':False})
		self.ee_ori = [-np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0]
		self._action_bound = 1.0
		self._ee_pos_scale = 0.02
		self._ee_ori_scale = np.pi / 36.0
		self._action_high = np.array([self._action_bound] * 2)
		self.action_space = spaces.Box(low=-self._action_high,
									   high=self._action_high,
									   dtype=np.float32)
		
		self.goal = np.array([0.6, -0.3, 1.0])
		self.init = np.array([0.6, 0.3, 1.0])
		self.robot.arm.reset()
		
		ori = euler2quat([0, 0, np.pi / 2])
		self.table_id = self.robot.pb_client.load_urdf('table/table.urdf',
													   [.5, 0, 0.4],
													   ori,
													   scaling=0.9)
		self.wall_id = self.robot.pb_client.load_geom('box', size=[0.10,0.01,0.2], mass=0,
													 base_pos=(0.5*self.goal + 0.5*self.init).tolist(),
													 rgba=[1, 0, 0, 0.4])
		self.marker_id = self.robot.pb_client.load_geom('box', size=0.05, mass=1,
													 base_pos=self.goal.tolist(),
													 rgba=[0, 1, 0, 0.4])
		# self.marker2_id = self.robot.pb_client.load_geom('box', size=0.03, mass=1,
		# 											 base_pos=[0.4, 0., 1.0],
		# 											 rgba=[0, 1, 0, 0.4])
		# self.marker3_id = self.robot.pb_client.load_geom('box', size=0.03, mass=0,
		# 												 base_pos=[0.8, 0.475, 1.0],
		# 												 rgba=[0, 0, 1, 0.4])
		# self.marker4_id = self.robot.pb_client.load_geom('box', size=0.03, mass=0,
		# 												 base_pos=[0.3, 0.125, 1.0],
		# 												 rgba=[0, 0, 1, 0.4])
		client_id = self.robot.pb_client.get_client_id()
		
		p.setCollisionFilterGroupMask(self.marker_id, -1, 0, 0, physicsClientId=client_id)
		p.setCollisionFilterPair(self.marker_id, self.table_id, -1, -1, 1, physicsClientId=client_id)

		self.reset()
		state_low = np.full(len(self._get_obs()), -float('inf'))
		state_high = np.full(len(self._get_obs()), float('inf'))
		self.observation_space = spaces.Box(state_low,
											state_high,
											dtype=np.float32)

	def reset(self):
		self.robot.arm.go_home(ignore_physics=True)
		jnt_pos = self.robot.arm.compute_ik(self.init)
		self.robot.arm.set_jpos(jnt_pos, ignore_physics=True)
		
		self.ref_ee_ori = self.robot.arm.get_ee_pose()[1]
		self.gripper_ori = 0
		self.timestep = 0
		return self._get_obs()

	def step(self, action):
		self.apply_action(action)
		state = self._get_obs()
		self.timestep += 1
		done = (self.timestep >= 200)
		info = dict()
		reward = self.compute_reward_reach_wall(state)
		return state, reward, done, info

	def compute_reward_reach_wall(self, state):
		# For naive approach:
		# l_two_goal = np.linalg.norm(self.goal - state, 2)
		# return -l_two_goal

		if state[1] > 0:
			penalty_point = np.array([0.8, 0.475, 1.0])
			l_two_penalty_point = np.linalg.norm(penalty_point - state, 2)
			tot_penalty = -0.5 * np.exp(-(l_two_penalty_point / (0.3 ** 2))) / 0.3

			# new_penalty_point = np.array([0.8, 0.125, 1.0])
			# newltwo = np.linalg.norm(new_penalty_point - state, 2)
			# whole_penalty = -0.5 * np.exp(-(newltwo / (0.3 ** 2))) / 0.3

			l_two_waypoint = np.linalg.norm(np.array([0.4, 0., 1.0]) - state, 2)
			reward = -l_two_waypoint

			frontside_reward = reward + tot_penalty
			return frontside_reward

		if state[1] < 0:
			penalty_point = np.array([0.3, 0.125, 1.0])
			l_two_penalty_point = np.linalg.norm(penalty_point - state, 2)
			tot_penalty = -2 * np.exp(-(l_two_penalty_point / (0.3 ** 2))) / 0.3

			l_two_goal = np.linalg.norm(self.goal - state, 2)
			goal_exp = np.exp(-5*l_two_goal ** 2)

			backstop_penalty = np.array([0.75, -0.5, 1.0])
			l_two_backstop = np.linalg.norm(backstop_penalty - state, 2)
			back_pen = -2 * np.exp(-(l_two_backstop / (0.3 ** 2))) / 0.3

			backside_reward = -4*l_two_goal + 2*goal_exp + tot_penalty + 3.65 + back_pen
			# print('backside reward:', backside_reward)
			return backside_reward

			# penalty_point = np.array([-0.2, 0., 1.0])
			# l_two_penalty_point = np.linalg.norm(penalty_point - state, 2)
			# wall_penalty = -np.exp(-(l_two_penalty_point / (0.7**2))) / 0.7
			# constant = -2
			# if state[1] > 0:
			# 	l_two_waypoint = np.linalg.norm(np.array([0.25, 0., 1.0]) - state, 2)
			# 	waypoint_reward = np.exp(constant * l_two_waypoint ** 2)
			# 	return waypoint_reward + wall_penalty
			# if state[1] < 0:
			# 	l_two_waypt_goal = np.linalg.norm(self.goal - np.array([0.25, 0., 1.0]), 2)
			# 	k = 1 - np.exp(constant * l_two_waypt_goal ** 2)
			# 	l_two_goal = np.linalg.norm(self.goal - state, 2)
			# 	goal_reward = np.exp(constant * l_two_goal ** 2) + k
			# 	return goal_reward + wall_penalty

	def _get_obs(self):
		gripper_pos = self.robot.arm.get_ee_pose()[0]
		return gripper_pos

	def apply_action(self, action):
		if not isinstance(action, np.ndarray):
			action = np.array(action).flatten()
		if action.size != 2:
			raise ValueError('Action should be [d_x, d_y].')

		action = np.concatenate([action, np.array([0.])])           
		pos, quat, rot_mat, euler = self.robot.arm.get_ee_pose()
		pos += action * self._ee_pos_scale

		rot_vec = np.array([0, 0, 1]) * self.gripper_ori
		rot_quat = rotvec2quat(rot_vec)
		ee_ori = quat_multiply(self.ref_ee_ori, rot_quat)
		jnt_pos = self.robot.arm.compute_ik(pos, ori=ee_ori)

		for step in range(self._action_repeat):
			self.robot.arm.set_jpos(jnt_pos)
			self.robot.pb_client.stepSimulation()

	def render(self, **kwargs):
		robot_base = self.robot.arm.robot_base_pos
		self.robot.cam.setup_camera(focus_pt=robot_base,
									dist=3,
									yaw=55,
									pitch=-30,
									roll=0)

		rgb, _ = self.robot.cam.get_images(get_rgb=True,
										   get_depth=False)
		return rgb

	def close(self):
		return
		